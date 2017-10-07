#!/usr/bin/env python

# SpeedyPP v2.0
# Last update: 25-01-2015 (AXP)

# Core Image Processing adapted from http://dx.doi.org/10.1016/j.neuroimage.2011.12.028
# Kundu, P., Inati, S.J., Evans, J.W., Luh, W.M. & Bandettini, P.A. Differentiating
#   BOLD and non-BOLD signals in fMRI time series using multi-echo EPI. NeuroImage (2012).

# De-noising module adapted for motion artifact removal by Patel, AX. (2014).
# http://dx.doi.org/10.1016/j.neuroimage.2014.03.012
# Patel, AX. et al. A wavelet method for modeling and despiking motion artifacts
#	from resting-state fMRI time series. NeuroImage (2014). 95:287-304.

# Script maintained and updated by Patel, AX. (2015). Email: ap531@cam.ac.uk

import sys
from re import split as resplit
import re
from os import system,getcwd,mkdir,chdir,popen
import os.path
from string import rstrip,split
from optparse import OptionParser,OptionGroup
import commands

#Filename parser for NIFTI and AFNI files
def dsprefix(idn):
	def prefix(datasetname):
		return split(datasetname,'+')[0]
	if len(split(idn,'.'))!=0:
		if split(idn,'.')[-1]=='HEAD' or split(idn,'.')[-1]=='BRIK' or split(idn,'.')[-2:]==['BRIK','gz']:
			return prefix(idn)
		elif split(idn,'.')[-1]=='nii' and not split(idn,'.')[-1]=='nii.gz':
			return '.'.join(split(idn,'.')[:-1])
		elif split(idn,'.')[-2:]==['nii','gz']:
			return '.'.join(split(idn,'.')[:-2])
		else:
			return prefix(idn)
	else:
		return prefix(idn)

def dssuffix(idna):
	suffix = idna.split(dsprefix(idna))[-1]
	spl_suffix=suffix.split('.')
	if len(spl_suffix[0])!=0 and spl_suffix[0][0] == '+':
		return spl_suffix[0]
	else:
		return suffix


#Configure options and help dialog
parser=OptionParser()
parser.add_option('-d',"",dest='dsinput',help="Input dataset. ex: -d PREFIX.nii.gz  ",default='')
parser.add_option('-f',"",dest='FWHM',help="Target smoothness (3dBlurToFWHM). ex: -f 8mm (default 5mm.) ",default='0mm')
parser.add_option('-a',"",dest='anat',help="Anatomical dataset. ex: -a mprage.nii.gz",default='')
parser.add_option('-o',"",action="store_true",dest='oblique',help="Oblique acqusition",default=False)

regropts=OptionGroup(parser,"Denoising (baseline regression) options, can specify multiple options. Requires anatomical.")
regropts.add_option('',"--rall", action="store_true",dest='rall',help="Regress all (except white matter).",default=False)
regropts.add_option('',"--rmot", action="store_true",dest='rmot',help="Regress motion.",default=False)
regropts.add_option('',"--rmotd", action="store_true",dest='rmotd',help="Regress motion derivatives.",default=False)
regropts.add_option('',"--rcsf", action="store_true",dest='rcsf',help="Regress CSF signal.",default=False)
regropts.add_option('',"--csfpeel", dest='csfpeel', help="Specify how harsh you would like the CSF masking from from 1 to 6, e.g. --csfpeel=6 (default=6)", default='6')
regropts.add_option('',"--csftemp", dest='csftemp', help="Specify CSF mask from MNI152, MNI_EPI, TT_MNI, TT_N27, TT_ICBM, TT_EPI, e.g. --csftemp MNInew (default=MNInew)",default='')
regropts.add_option('',"--rwm", action="store_true",dest='rwm',help="Regress white matter signal (not recommended).",default=False)
parser.add_option_group(regropts)

freqopts=OptionGroup(parser,"Frequency filtering options")
freqopts.add_option('',"--lowpass",dest='lowpass',help="Low pass frequencey in Hz, default=0.1",default='0.1')
freqopts.add_option('',"--highpass",dest='highpass',help="Highpass filter in Hz, default=0.01",default='0.01')
freqopts.add_option('',"--nobandpass",action="store_true",help="Keep all frequencies except Nyquist and zero frequencies. This option overwrites any specified frequency bands. Use this if you intend to perform wavelet filtering, otherwise please specify a Fourier filter.",default=False)
parser.add_option_group(freqopts)

extopts=OptionGroup(parser,"Extended preprocessing options")
# Denoising options
extopts.add_option('',"--tds",action="store_true",dest='tds',help="Applies a temporal despike to data. --wds is a better alternative.",default=False)
extopts.add_option('',"--wds",action="store_true",dest='wds',help="Applies Wavelet Despiking to data.",default=False)
extopts.add_option('',"--threshold",dest='threshold',help="Wavelet Despiking threshold initiator variable.",default='10')
extopts.add_option('',"--SP",action="store_true",dest='SP',help="Output Spike Percentage (can only be used with --wds)",default=False)
#extopts.add_option('',"--EDOF",action="store_true",dest='EDOF',help="Output Degrees of freedom (can only be used with --wds)",default=False)
#extopts.add_option('',"--TW",action="store_true",dest='TW',help="Use Wavelet Despiking to compute EDOF-constant Time windows.",default=False)
#extopts.add_option('',"--window_init",dest='window_init',help="Specify initial window length for TW analysis.",default='100')
# Extended input specifications
extopts.add_option('',"--basetime",dest='basetime',help="Time to steady-state equilibration in seconds. Default 0. ex: --basetime 10 ",default=0)
extopts.add_option('',"--betmask",dest='betmask',action='store_true',help="More lenient functional masking (using BET).",default=False)
extopts.add_option('',"--skullstrip",action="store_true",dest='skullstrip',help="Skullstrip anatomical",default=False)
extopts.add_option('',"--ss",dest='ss',help="Transform brain to standard space. Options are MNI152, MNI_caez, TT_MNI, TT_N27, TT_ICBM, e.g. --ss TT_MNI",default='')
extopts.add_option('',"--qwarp",dest='qwarp',help="Use non-linear warping for standard space transform, e.g. --qwarp",default=False)
extopts.add_option('',"--align_ss",action="store_true",dest='align_ss',help="Try this flag if standard space transform has failed",default=False)
extopts.add_option('',"--coreg_cfun",dest='coreg_cfun',help="specify coreg cost functional. Default lpc. ex: --coreg_cfun=nmi",default='lpc')
extopts.add_option('',"--TR",dest='TR',help="The TR. Default is to read from input dataset",default='')
extopts.add_option('',"--tpattern",dest='tpattern',help="Slice timing (i.e. alt+z, see 3dTshift --help). Default from header. Correction skipped if not found.",default='')
extopts.add_option('',"--zeropad",dest='zeropad',help="Zeropadding options. -z N means add N slabs in all directions. Default 15 (N.B. autoboxed after coregistration)",default="15")
extopts.add_option('',"--keep_means",dest='keep_means',action="store_true",help="output fully pre-processed NifTI file with means still in time series instead of de-meaned default counterpart.",default=False)
extopts.add_option('',"--align_base",dest='align_base',help="Base EPI for allineation",default='')
extopts.add_option('',"--align_interp",dest='align_interp',help="Interpolation method for allineation",default='cubic')
extopts.add_option('',"--align_args",dest='align_args',help="Additional arguments for 3dAllineate EPI-anatomical alignment",default='')
extopts.add_option('',"--label",dest='label',help="Extra label to tag this SpeedyPP folder",default='')
extopts.add_option('',"--keep_int",action="store_true",dest='keep_int',help="Keep preprocessing intermediates. Default is delete to save space.",default=False)
extopts.add_option('',"--OVERWRITE",dest='overwrite',action="store_true",help="If spp.xyz directory exists, overwrite. ",default=False)
extopts.add_option('',"--CLUSTER",dest='cluster',action="store_true",help="Cluster mode (disables greedy multithreading). ",default=False)
extopts.add_option('',"--exit",action="store_true",dest='exit',help="Generate script and exit",default=0)
parser.add_option_group(extopts)

(options,args) = parser.parse_args()

#Parse dataset input names and option pair errors
if options.dsinput=='':
	print "Need at least input dataset. Try speedypp.py -h"
	sys.exit()
if os.path.abspath(os.path.curdir).__contains__('spp.'):
	print "You are inside a SpeedyPP directory! Please leave this directory and rerun."
	sys.exit()
if options.ss and options.anat=='':
	print "You specified standard space normalization without an anatomical reference. This is disabled, exiting."
	sys.exit()
if (options.rwm or options.rcsf or options.rall) and options.anat=='':
	print "You specified WM or CSF regression without an anatomical reference. Exiting."
	sys.exit()
if options.tds and options.wds:
	print "You cannot specify both Time and Wavelet Despike methods."
	sys.exit()
if (options.SP or options.threshold) and not options.wds:
	print "Please specify --wds with other Wavelet Despiking options. Exiting."
	sys.exit()
if options.threshold and not options.wds:
	print "You have specified a Wavelet Despiking initial threshold, but not the --wds flag. Please try again with --wds. Exiting"
	sys.exit()

#if (options.csfpeel or options.csftemp) and not (options.rall or options.rcsf):
#	print "You have specified a csf template and mask peel option but not csf regression. Exiting."
#	sys.exit()

#Parse prefix and suffix
dsinput=dsprefix(options.dsinput)
prefix=dsprefix(options.dsinput)
isf =dssuffix(options.dsinput)

#Parse timing arguments
if options.TR!='' : tr=float(options.TR)
else:
	#print '3dinfo -tr %s%s' % (prefix,isf)
	tr=float(os.popen('3dinfo -tr %s%s' % (prefix,isf)).readlines()[0].strip())
	options.TR=str(tr)
timetoclip=float(options.basetime)
basebrik=int(timetoclip/tr)

if options.nobandpass:
	lowpass=float('99999')
	highpass=float('0')
else:
	highpass = float(options.highpass)
	lowpass = float(options.lowpass)


#Prepare script variables
sppdir=os.path.dirname(sys.argv[0])
sl = []			#Script command list
sl.append('#'+" ".join(sys.argv).replace('"',r"\""))

osf='.nii.gz'		#Using NIFTI outputs

vrbase=prefix
if options.align_base == '':
	align_base = basebrik
else:
	align_base = options.align_base
setname=prefix+options.label
startdir=rstrip(popen('pwd').readlines()[0])
combstr=""; allcombstr=""

if options.overwrite:
	system('rm -rf spp.%s' % (setname))
else:
	sl.append("if [[ -e spp.%s/stage ]]; then echo SpeedyPP directory exists, exiting \(try --OVERWRITE\).; exit; fi" % (setname))

sl.append("starttime=`date +%s`")

system('mkdir -p spp.%s' % (setname))

sl.append("cp _spp_%s.sh spp.%s/" % (setname,setname))
sl.append("cd spp.%s" % setname)
thecwd= "%s/spp.%s" % (getcwd(),setname)

if len(split(options.zeropad))==1 :
	zeropad_opts=" -I %s -S %s -A %s -P %s -L %s -R %s " % (tuple([options.zeropad]*6))
elif options.zeropad!='':
	zeropad_opts=options.zeropad
else:
	zeropad_opts=''

# Time Despike options
if options.tds:
	despike_opt = "-despike"
else:
	despike_opt = ""
if options.cluster: sl.append("export OMP_NUM_THREADS=1")

# Parse strings for csf masking and standard space transform
if options.anat!='':
	if options.rcsf or options.rall or options.ss!='':
		tempdir="$HOME/fmri_spt/templates"
		if options.csftemp=="MNI152" or options.csftemp=="mni152":
			tempstr="MNI152_T1_1mm+tlrc"
			sl.append("csfstr=MNI152_T1_1mm_csf_mask%s" % (str(options.csfpeel)))
		elif options.csftemp=="MNI_caez" or options.csftemp=="mni_caez":
			tempstr="MNI_caez_N27+tlrc"
			sl.append("csfstr=MNI_caez_csf_mask%s" % (str(options.csfpeel)))
		elif options.csftemp=="MNI_EPI" or options.csftemp=="mni_epi":
			tempstr="MNI_EPI+tlrc"
			sl.append("csfstr=MNI_EPI_csf_mask%s" % (str(options.csfpeel)))
		elif options.csftemp=="TT_MNI" or options.csftemp=="tt_mni":
			tempstr="TT_avg152T1+tlrc"
			sl.append("csfstr=TT_avg152T1_csf_mask%s" % (str(options.csfpeel)))
		elif options.csftemp=="TT_N27" or options.csftemp=="tt_n27":
			tempstr="TT_N27+tlrc"
			sl.append("csfstr=TT_N27_csf_mask%s" % (str(options.csfpeel)))
		elif options.csftemp=="TT_ICBM" or options.csftemp=="tt_icbm":
			tempstr="TT_icbm452+tlrc"
			sl.append("csfstr=TT_icbm452_csf_mask%s" % (str(options.csfpeel)))
		elif options.csftemp=='':
			tempstr="MNI152_T1_1mm+tlrc"
			sl.append("csfstr=MNI152_T1_1mm_csf_mask%s" % (str(options.csfpeel)))
		else:
			print "I do not understand the csftemp template entered. Options are: MNIL MNIS N27 ICBM and EPI."
			sys.exit()

	if options.ss!='':
		if options.ss=="MNI152" or options.ss=="mni152":
			ssstr="MNI152_T1_1mm+tlrc"
		elif options.ss=="MNI_caez" or options.ss=="mni_caez":
			ssstr="MNI_caez_N27+tlrc"
		elif options.ss=="TT_MNI" or options.ss=="tt_mni":
			ssstr="TT_avg152T1+tlrc"
		elif options.ss=="TT_N27" or options.ss=="tt_n27":
			ssstr="TT_N27+tlrc"
		elif options.ss=="TT_ICBM" or options.ss=="tt_icbm":
			ssstr="TT_icbm452+tlrc"
		else:
			print "Cannot understand the standard space template entered."
			sys.exit()

	if options.ss=='' and options.qwarp:
		print "You have specified non-linear standard space warping with qwarp, but no standard space template."
		print "Please specify the --ss option on input. Exiting."
		sys.exit()

	if options.ss!='' and (options.rcsf or options.rall):
		if tempstr!=ssstr and options.csftemp!='':
			tempstr=ssstr
			print "The csf and standard space templates do not match. Using ss template for csf masking and standard space warp."
		else:
			tempstr=ssstr

	if options.align_ss and not options.ss:
		print "You have specified align_ss without specifying standard space warp. This option is disabled. Exiting."
		sys.exit()


if str(options.csfpeel)!='6' and str(options.csfpeel)!='5' and str(options.csfpeel)!='4' and str(options.csfpeel)!='3' and str(options.csfpeel)!='2' and str(options.csfpeel)!='1':
	print "Please specify a csfpeel between 1 and 3. Default=6. Exiting"
	sys.exit()


#########################
#########################

# Pre-processing preamble

print """\n ** SpeedyPP pre-processing script for resting-state fMRI! **

Core Image Processing Module: Prantik Kundu (2012). See http://dx.doi.org/10.1016/j.neuroimage.2011.12.028.
De-Noising Module:  Ameera X Patel (2014). See http://dx.doi.org/10.1016/j.neuroimage.2014.03.012

Please cite:

Patel, AX. et al. A wavelet method for modeling and despiking motion artifacts
from resting-state fMRI time series. NeuroImage (2014). 95:287-304.

"""


#Parse anatomical processing options, process anatomical
if options.anat != '':
	nsmprage = options.anat
	anatprefix=dsprefix(nsmprage)
	if options.align_ss:
		sl.append("3dcopy -overwrite %s/%s %s/%s.al" % (startdir,nsmprage,startdir,anatprefix))
		sl.append("@align_centers -no_cp -dset %s/%s.al+orig -base %s/standard/%s" % (startdir,anatprefix,tempdir,ssstr))
		nsmprage = anatprefix + '.al' + '+orig'
	pathanatprefix="%s/%s" % (startdir,anatprefix)
	if options.oblique:
		sl.append("if [ ! -e %s_sppdo.nii.gz ]; then 3dWarp -overwrite -prefix %s_sppdo.nii.gz -deoblique %s/%s; fi" % (pathanatprefix,pathanatprefix,startdir,nsmprage))
		nsmprage="%s_sppdo.nii.gz" % (anatprefix)
	if options.skullstrip:
		sl.append("if [ ! -e %s_ns.nii.gz ]; then 3dUnifize -overwrite -prefix %s_u.nii.gz %s/%s; 3dSkullStrip  -shrink_fac_bot_lim 0.3 -orig_vol -overwrite -prefix %s_ns.nii.gz -input %s_u.nii.gz; 3dAutobox -overwrite -prefix %s_ns.nii.gz %s_ns.nii.gz; fi" % (pathanatprefix,pathanatprefix,startdir,nsmprage,pathanatprefix,pathanatprefix,pathanatprefix,pathanatprefix))
		nsmprage="%s_ns.nii.gz" % (anatprefix)

# Calculate rigid body alignment
vrAinput = "%s/%s%s" % (startdir,vrbase,isf)
sl.append("3dvolreg -tshift -quintic -prefix ./%s_vrA%s -base %s[%s] -dfile ./%s_vrA.1D -1Dmatrix_save ./%s_vrmat.aff12.1D %s" % (vrbase,isf,vrAinput,basebrik,vrbase,prefix,vrAinput))
vrAinput = "./%s_vrA%s" % (vrbase,isf)
if options.oblique:
	if options.anat!='':
		sl.append("3dWarp -verb -card2oblique %s[%s] -overwrite  -newgrid 1.000000 -prefix ./%s_ob.nii.gz %s/%s | \grep  -A 4 '# mat44 Obliquity Transformation ::'  > %s_obla2e_mat.1D" % (vrAinput,basebrik,anatprefix,startdir,nsmprage,prefix))
	else:
		sl.append("3dWarp -overwrite -prefix %s -deoblique %s" % (vrAinput,vrAinput))
sl.append("1dcat './%s_vrA.1D[1..6]{%s..$}' > %s/%s_motion.1D " % (vrbase,basebrik,startdir,prefix))


# Compute Tshift and mask
dsin = prefix
if options.tpattern!='':
	tpat_opt = ' -tpattern %s ' % options.tpattern
else:
	tpat_opt = ''
sl.append("3dTshift -heptic %s -prefix ./%s_ts+orig %s/%s%s" % (tpat_opt,dsin,startdir,prefix,isf) )
if options.oblique and options.anat=="":
	sl.append("3dWarp -overwrite -deoblique -prefix ./%s_ts+orig ./%s_ts+orig" % (dsin,dsin))
sl.append("3drefit -deoblique -TR %s %s_ts+orig" % (options.TR,dsin))
sl.append("3dAllineate -overwrite -final NN -NN -float -1Dmatrix_apply %s_vrmat.aff12.1D'{%i..%i}' -base %s[%s] -input %s_ts+orig'[%i..%i]' -prefix %s_vrA.nii.gz" % \
				(prefix,int(basebrik),int(basebrik)+5,vrAinput,basebrik,dsin,int(basebrik),int(basebrik)+5,dsin))


sl.append("3dcalc -expr 'a' -a %s[%s] -prefix ./_eBmask%s" % (vrAinput,align_base,osf))
sl.append("bet _eBmask%s eBmask%s" % (osf,osf))
sl.append("fast -t 2 -n 3 -H 0.1 -I 4 -l 20.0 -b -o eBmask eBmask%s" % (osf))
sl.append("3dcalc -a eBmask%s -b eBmask_bias%s -expr 'a/b' -prefix eBbase%s" % ( osf, osf, osf))

# Calculate affine anatomical warp if anatomical provided, then combine motion correction and coregistration parameters
if options.anat!='':
	sl.append("cp %s/%s* ." % (startdir,nsmprage))

	if options.ss:
		atnsmprage = "%s_at.nii.gz" % (dsprefix(nsmprage))
		if not dssuffix(nsmprage).__contains__('nii'):
			sl.append("3dcalc -a %s -expr 'a' -prefix %s.nii.gz" % (nsmprage,dsprefix(nsmprage)))
		sl.append("if [ ! -e %s ]; then \@auto_tlrc -no_ss -base %s/standard/%s -input %s.nii.gz -suffix _at -ok_notice; fi " % (atnsmprage,tempdir,ssstr,dsprefix(nsmprage)) )
		sl.append("gzip -f ./%s.nii" % (dsprefix(atnsmprage)) )
		sl.append("3dAutobox -prefix ./sstemp.nii.gz %s/standard/%s" % (tempdir,ssstr) )
#		atnsmprage = "%s_at.nii.gz" % (dsprefix(nsmprage))
		if options.qwarp:
			sl.append("if [ ! -e %s/%snl.nii.gz ]; then " % (startdir,dsprefix(atnsmprage)) )
			sl.append("3dUnifize -overwrite -GM -prefix ./%su.nii.gz ./%s" % (dsprefix(atnsmprage),atnsmprage) )
			sl.append("3dQwarp -overwrite -resample -duplo -useweight -blur 2 2 -workhard -base %s/standard/%s -prefix ./%snl.nii.gz -source ./%su.nii.gz" % (tempdir,ssstr,dsprefix(atnsmprage),dsprefix(atnsmprage)) )
			sl.append("3dcopy ./%snl.nii.gz %s/" % (dsprefix(atnsmprage),startdir) )
			sl.append("fi")

	align_args=""
	if options.align_args!="":
		align_args=options.align_args
	elif options.oblique:
		align_args = " -cmass -maxrot 30 -maxshf 30 "
	else:
		align_args=" -maxrot 20 -maxshf 20 -parfix 7 1  -parang 9 0.83 1.0 "
	if options.oblique:
		alnsmprage = "./%s_ob.nii.gz" % (anatprefix)
	else:
		alnsmprage = "%s/%s" % (startdir,nsmprage)
	coreg_cfun = '-%s' % options.coreg_cfun
	# sl.append("3dAllineate -weight_frac 1.0 -VERB -warp aff -weight eBmask_pve_0.nii.gz -lpc -base eBbase.nii.gz -master SOURCE -source %s -prefix ./%s_al -1Dmatrix_save %s_al_mat %s" % (alnsmprage, anatprefix,anatprefix,align_args))
	sl.append("3dAllineate -weight_frac 1.0 -VERB -warp aff -weight eBmask_pve_0.nii.gz %s -base eBbase.nii.gz -master SOURCE -source %s -prefix ./%s_al -1Dmatrix_save %s_al_mat %s" % (coreg_cfun, alnsmprage, anatprefix,anatprefix,align_args))
	if options.ss:
		tlrc_opt = "%s::WARP_DATA -I" % (atnsmprage)
	else:
		tlrc_opt = ""
	if options.oblique:
		oblique_opt = "%s_obla2e_mat.1D" % prefix
	else:
		oblique_opt = ""
	sl.append("cat_matvec -ONELINE  %s %s %s_al_mat.aff12.1D -I  %s_vrmat.aff12.1D  > %s_wmat.aff12.1D" % (tlrc_opt,oblique_opt,anatprefix,prefix,prefix))
else:
	sl.append("cp %s_vrmat.aff12.1D %s_wmat.aff12.1D" % (prefix,prefix))

#Detect if current AFNI has old 3dNwarpApply
if " -affter aaa  = *** THIS OPTION IS NO LONGER AVAILABLE" in commands.getstatusoutput("3dNwarpApply -help")[1]:
	old_qwarp = False
else:
	old_qwarp = True

# Extended Image Processing

if zeropad_opts!="" :
	sl.append("3dZeropad %s -prefix _eBvrmask.nii.gz %s_ts+orig[%s]" % (zeropad_opts,dsin,basebrik))

if options.anat!='':
	if options.qwarp:
		sl.append("voxsize=`ccalc $(3dinfo -voxvol _eBvrmask.nii.gz)**.33`") #Set voxel size
		if old_qwarp:
			sl.append("3dNwarpApply -overwrite -nwarp './%snl_WARP.nii.gz' -affter '%s_wmat.aff12.1D' -master sstemp.nii.gz -dxyz ${voxsize} -source _eBvrmask.nii.gz -interp NN -prefix ./_eBvrmask.nii.gz" % (dsprefix(atnsmprage),prefix))
		else:
			sl.append("3dNwarpApply -overwrite -nwarp './%snl_WARP.nii.gz' '%s_wmat.aff12.1D' -master sstemp.nii.gz -dxyz ${voxsize} -source _eBvrmask.nii.gz -interp NN -prefix ./_eBvrmask.nii.gz" % (dsprefix(atnsmprage),prefix))
		if options.betmask:
			sl.append("bet _eBvrmask%s eBvrmask%s " % (osf,osf ))
		else:
			sl.append("3dAutomask -overwrite -prefix eBvrmask%s _eBvrmask%s" % (osf,osf))
		sl.append("3dAutobox -overwrite -prefix eBvrmask%s eBvrmask%s" % (osf,osf) )
		sl.append("3dcalc -a eBvrmask.nii.gz -expr 'notzero(a)' -overwrite -prefix eBvrmask.nii.gz")
		if old_qwarp:
			sl.append("3dNwarpApply -nwarp './%snl_WARP.nii.gz' -affter %s_wmat.aff12.1D -master eBvrmask.nii.gz -source %s_ts+orig -interp %s -prefix ./%s_vr%s " % (dsprefix(atnsmprage),prefix,dsin,options.align_interp,dsin,osf) )
		else:
			sl.append("3dNwarpApply -nwarp './%snl_WARP.nii.gz' %s_wmat.aff12.1D -master eBvrmask.nii.gz -source %s_ts+orig -interp %s -prefix ./%s_vr%s " % (dsprefix(atnsmprage),prefix,dsin,options.align_interp,dsin,osf) )
	else:
		sl.append("3dAllineate -overwrite -final %s -%s -float -1Dmatrix_apply %s_wmat.aff12.1D -base _eBvrmask.nii.gz -input _eBvrmask.nii.gz -prefix ./_eBvrmask.nii.gz" % \
			(options.align_interp,options.align_interp,prefix))
		if options.betmask:
			sl.append("bet _eBvrmask%s eBvrmask%s " % (osf,osf ))
		else:
			sl.append("3dAutomask -overwrite -prefix eBvrmask%s _eBvrmask%s" % (osf,osf))
		sl.append("3dAutobox -overwrite -prefix eBvrmask%s eBvrmask%s" % (osf,osf) )
		sl.append("3dcalc -a eBvrmask.nii.gz -expr 'notzero(a)' -overwrite -prefix eBvrmask.nii.gz")
		sl.append("3dAllineate -final %s -%s -float -1Dmatrix_apply %s_wmat.aff12.1D -base eBvrmask%s -input  %s_ts+orig -prefix ./%s_vr%s" % \
			(options.align_interp,options.align_interp,prefix,osf,dsin,dsin,osf))
else:
	sl.append("3dAutobox -overwrite -prefix eBvrmask%s eBvrmask%s" % (osf,osf) )
	sl.append("3dcalc -a eBvrmask.nii.gz -expr 'notzero(a)' -overwrite -prefix eBvrmask.nii.gz")


if options.FWHM=='0mm' or str(options.FWHM)=='0':
	sl.append("3dcalc -prefix ./%s_sm%s -a ./%s_vr%s[%i..$] -b eBvrmask.nii.gz -expr 'a*notzero(b)' " % (dsin,osf,dsin,osf,basebrik))
else:
	sl.append("3dBlurInMask -fwhm %s -mask eBvrmask%s -prefix ./%s_sm%s ./%s_vr%s[%i..$]" % (options.FWHM,osf,dsin,osf,dsin,osf,basebrik))
sl.append("3dBrickStat -mask eBvrmask.nii.gz -percentile 50 1 50 %s_sm%s[%s] > gms.1D" % (dsin,osf,basebrik))
sl.append("gms=`cat gms.1D`; gmsa=($gms); p50=${gmsa[1]}")
sl.append("3dcalc -overwrite -a ./%s_sm%s -expr \"a*1000/${p50}\" -prefix ./%s_sm%s" % (dsin,osf,dsin,osf))
#sl.append("3dTstat -prefix ./%s_mean%s ./%s_sm%s" % (dsin,osf,dsin,osf))

# Denoising #
sl.append("echo")
sl.append("echo -----------------------------------------")
sl.append("echo *+*+* Initialising Denoising Module *+*+*")
sl.append("echo -----------------------------------------")

if options.wds:
	if options.SP:
		sl.append("matlab -nodisplay -r \"WaveletDespike('%s_sm%s','%s','SP',1,'threshold',%s);quit\" " % (dsin,osf,dsin,float(options.threshold)))
	else:
		sl.append("matlab -nodisplay -r \"WaveletDespike('%s_sm%s','%s','SP',0,'threshold',%s);quit\" " % (dsin,osf,dsin,float(options.threshold)))
	sl.append("3dcopy %s_wds%s %s_in%s" % (dsin,osf,dsin,osf))
#	sl.append("3dBandpass -prefix ./%s_in%s 0 99999 ./%s_wds.nii.gz " % (dsin,osf,dsin))
	if options.ss:
		if ssstr=="MNI152_T1_1mm+tlrc" or ssstr=="MNI_caez_N27+tlrc":
			sl.append("3drefit -space MNI %s_noise%s" % (dsin,osf))
		else:
			sl.append("3drefit -space TLRC %s_noise%s" % (dsin,osf))
else:
	sl.append("3dcopy %s_sm%s %s_in%s" % (dsin,osf,dsin,osf))
#sl.append("3dBandpass %s -prefix ./%s_in%s 0 99999 ./%s_sm%s " % (despike_opt,dsin,osf,dsin,osf))

#sl.append("3dcalc -overwrite -a ./%s_in%s -b ./%s_mean%s -expr 'a+b' -prefix ./%s_in%s" % (dsin,osf,dsin,osf,dsin,osf))
#sl.append("3dTstat -stdev -prefix ./%s_std%s ./%s_in%s" % (dsin,osf,dsin,osf))

if options.ss:
	if ssstr=="MNI152_T1_1mm+tlrc" or ssstr=="MNI_caez_N27+tlrc":
		sl.append("3drefit -space MNI %s_in%s" % (dsin,osf))
	else:
		sl.append("3drefit -space TLRC %s_in%s" % (dsin,osf))

#if not options.keep_int: sl.append("rm %s_ts+orig* %s_vr%s %s_sm%s" % (dsin,dsin,osf,dsin,osf)) #THIS WAS THE ORIGINAL!
if not options.keep_int:
	sl.append("rm %s_ts+orig* %s_vr%s" % (dsin,dsin,osf))

#Build denoising regressors
regs = []
if options.rall or options.rmot:
	sl.append("1d_tool.py -demean -infile %s/%s_motion.1D -write motion_dm.1D" % (startdir,prefix))
	regs.append("motion_dm.1D")
if options.rall or options.rmotd:
	sl.append("1d_tool.py -demean -derivative -infile %s/%s_motion.1D -write motion_deriv.1D" % (startdir,prefix))
	regs.append("motion_deriv.1D")
if (options.rall or options.rmot or options.rmotd):
	sl.append("1dcat %s > %s_reg_baseline_pre.1D " % (' '.join(regs),prefix))
	sl.append("3dBandpass -ort %s_reg_baseline_pre.1D -prefix ./%s_pppre.nii.gz %s 99999 ./%s_in.nii.gz" % (prefix,prefix,highpass,prefix))
else:
	sl.append("3dBandpass -prefix ./%s_pppre.nii.gz %s 99999 ./%s_in.nii.gz" % (prefix,highpass,prefix))

if (options.rall or options.rcsf or options.rwm):
	sl.append("echo Downsampling anatomical and segmenting with FSL FAST...")
	if options.ss and options.qwarp:
		sl.append("3dresample -rmode Li -master eBvrmask.nii.gz -inset %snl.nii.gz -prefix %s_epi.nii.gz" % (dsprefix(atnsmprage),dsprefix(nsmprage)))
	elif options.ss and not options.qwarp:
		sl.append("3dresample -rmode Li -master eBvrmask.nii.gz -inset %s.nii.gz -prefix %s_epi.nii.gz" % (dsprefix(atnsmprage),dsprefix(nsmprage)))
	else:
		sl.append("3dresample -rmode Li -master eBvrmask.nii.gz -inset %s -prefix %s_epi.nii.gz" % (nsmprage,dsprefix(nsmprage)))
	sl.append("fast -t 1 -n 3 -H 0.1 -I 4 -l 20.0 -g -b -o %s_epi %s_epi.nii.gz" % (dsprefix(nsmprage),dsprefix(nsmprage)))
	if options.rall or options.rcsf:
		if options.ss:
			sl.append("3dresample -rmode NN -master eBvrmask.nii.gz -inset %s/mask/${csfstr}.nii.gz -prefix ./${csfstr}_%s_native.nii.gz" % (tempdir,prefix))
			if str(options.csfpeel)=='1' or str(options.csfpeel)=='3' or str(options.csfpeel)=='5':
				sl.append("3dcalc -a %s_epi_pve_0.nii.gz -b ${csfstr}_%s_native.nii.gz -expr 'notzero(b)*within(a,0.75,1)' -prefix %s_csfmask.nii.gz" % (dsprefix(nsmprage),prefix,prefix))
			elif str(options.csfpeel)=='2' or str(options.csfpeel)=='4' or str(options.csfpeel)=='6':
				sl.append("3dcalc -a %s_epi_pve_0.nii.gz -b ${csfstr}_%s_native.nii.gz -expr 'notzero(b)*equals(a,1)' -prefix %s_csfmask.nii.gz" % (dsprefix(nsmprage),prefix,prefix))
		else:
			sl.append("\@auto_tlrc -suffix _at -base %s/standard/%s -input %s -no_ss -ok_notice" % (tempdir,tempstr,nsmprage))
			sl.append("if [ ! -e %s/spp.%s/%s_at.Xat.1D ]; then rm %s_XYZ* __ats_temp*; tempstr='TT_N27+tlrc'; csfstr='TT_N27_csf_mask%s'; \@auto_tlrc -suffix _at -base %s/standard/${tempstr} -input %s -no_ss -ok_notice; fi" % (startdir,setname,dsprefix(nsmprage),dsprefix(nsmprage),str(options.csfpeel),tempdir,nsmprage))
			sl.append("cat_matvec %s_at.Xat.1D -I -ONELINE > %s_at_inv.1D" % (dsprefix(nsmprage),dsprefix(nsmprage)))
			sl.append("3dAllineate -1Dmatrix_apply %s_at_inv.1D -base %s_in.nii.gz -input %s/mask/${csfstr}.nii.gz -prefix ${csfstr}_%s_native.nii.gz -interp NN" % (dsprefix(nsmprage),prefix,tempdir,prefix))
			if str(options.csfpeel)=='1' or str(options.csfpeel)=='3' or str(options.csfpeel)=='5':
				sl.append("3dcalc -a %s_epi_pve_0.nii.gz -b ${csfstr}_%s_native.nii.gz -expr 'notzero(b)*within(a,0.75,1)' -prefix %s_csfmask.nii.gz" % (dsprefix(nsmprage),prefix,prefix))
			elif str(options.csfpeel)=='2' or str(options.csfpeel)=='4' or str(options.csfpeel)=='6':
				sl.append("3dcalc -a %s_epi_pve_0.nii.gz -b ${csfstr}_%s_native.nii.gz -expr 'notzero(b)*equals(a,1)' -prefix %s_csfmask.nii.gz" % (dsprefix(nsmprage),prefix,prefix))
		sl.append("3dcalc -overwrite -prefix %s_csfmask.nii.gz -a %s_csfmask.nii.gz -b eBvrmask.nii.gz -expr 'a*b'" % (prefix,prefix))
		sl.append("3dresample -overwrite -prefix %s_csfmask.nii.gz -master %s_pppre.nii.gz -input %s_csfmask.nii.gz" % (prefix,prefix,prefix))
		sl.append("3dmaskave -quiet -mask %s_csfmask.nii.gz %s_pppre.nii.gz > %s_csf.1D" % (prefix,prefix,prefix))
		sl.append("if [ ! -e %s_csf.1D ] && [ %s -ge 2 ]; then echo The csf mask was not generated. You used csfpeel=%s. Try again with a less harsh csfpeel; elif [ ! -e %s_csf.1D ] && [ %s -eq 1 ]; then echo Sometshing went wrong when generating the csf mask; fi" % (prefix,options.csfpeel,options.csfpeel,prefix,options.csfpeel))
		regs.append("%s_csf.1D" % (prefix))
	if options.rwm:
		sl.append("3dcalc -a %s_epi_seg.nii.gz -b eBvrmask.nii.gz -expr 'notzero(b)*(equals((a+b),4))' -prefix %s_wmmask.nii.gz" % (dsprefix(nsmprage),prefix) )
		sl.append("3dresample -overwrite -prefix %s_wmmask.nii.gz -master %s_pppre.nii.gz -input %s_wmmask.nii.gz" % (prefix,prefix,prefix))
		sl.append("3dmaskave -quiet -mask %s_wmmask.nii.gz %s_pppre.nii.gz > %s_wm.1D" % (prefix,prefix,prefix))
		regs.append("%s_wm.1D" % (prefix))

if regs!=[]:
	sl.append("1dcat %s > %s_reg_baseline.1D" % (' '.join(regs), prefix))
	if options.keep_means:
		sl.append("3dTstat -prefix ./%s_meanin%s ./%s_in%s" % (dsin,osf,dsin,osf))
		sl.append("3dBandpass -ort %s_reg_baseline.1D -prefix ./%s_pp%s %f %f ./%s_in%s" % (prefix,prefix,osf,highpass,lowpass,prefix,osf))
		sl.append("3dcalc -overwrite -a ./%s_pp%s -b ./%s_meanin%s -expr 'a+b' -prefix ./%s_pp%s" % (dsin,osf,dsin,osf,dsin,osf))
	else:
		sl.append("3dBandpass -ort %s_reg_baseline.1D -prefix ./%s_pp%s %f %f ./%s_in%s" % (prefix,prefix,osf,highpass,lowpass,prefix,osf))

# Mask out extra non-brain voxels
sl.append("3dcalc -prefix mask%s -expr 'bool(a)' -a %s_sm%s" % (osf,prefix,osf))
if options.ss:
	if ssstr=="MNI152_T1_1mm+tlrc" or ssstr=="MNI_EPI+tlrc":
		sl.append("3drefit -space MNI mask%s" % (osf))
	else:
		sl.append("3drefit -space TLRC mask%s" % (osf))
sl.append("3dresample -overwrite -prefix mask%s -master %s_in%s -input mask%s" % (osf,prefix,osf,osf))
sl.append("3dcalc -overwrite -prefix %s_in%s -expr 'a*b' -a %s_in%s -b mask%s" % (prefix,osf,prefix,osf,osf))
if regs!=[]:
	sl.append("3dcalc -overwrite -prefix %s_pp%s -expr 'a*b' -a %s_pp%s -b mask%s" % (prefix,osf,prefix,osf,osf))

#	if options.ss:
#		if ssstr=="MNI152_T1_1mm+tlrc" or ssstr=="MNI_EPI+tlrc":
#			sl.append("3drefit -space MNI %s_pp%s" % (prefix,osf))
#		else:
#			sl.append("3drefit -space TLRC %s_pp%s" % (prefix,osf))

#Copy processed files into start directory, compute time taken, and feedback to user
if regs!=[]:
	if options.ss:
		sl.append("cp %s_pp.nii.gz %s_in.nii.gz %s %s" % (prefix,prefix,atnsmprage,startdir))
	else:
		sl.append("cp %s_pp.nii.gz %s_in.nii.gz %s" % (prefix,prefix,startdir))
else:
	sl.append("cp %s_in.nii.gz %s" % (prefix,startdir))

if options.SP:
	sl.append("cp %s_SP.txt %s" % (prefix,startdir))

sl.append("stoptime=`date +%s`")
sl.append("if [[ $tempstr == *TT_N27* ]]; then echo; echo Warping for CSF masking failed to converge with the MNI template. The talairach N27 template was used instead. You may find this the case when warping %s_pp.nii.gz to standard space; fi" % (prefix) )
if options.rcsf or options.rall:
	sl.append("echo; echo Please perform the following visual checks on your data; echo; echo 1. check co-registration of %s_pp.nii.gz to %s; echo 2. check to see no initial positive transients detected; echo 3. check CSF mask, spp.%s/%s_csfmask.nii.gz, for alignment and quality;" % (prefix,nsmprage,setname,prefix) )
#else:
#	sl.append("echo; echo Please perform the following visual checks on your data; echo; echo 1. check co-registration of %s_pp.nii.gz to %s; echo 2. check to see no initial positive transients detected" % (prefix,nsmprage,setname,prefix) )

sl.append("echo; echo; echo Preprocessing required `ccalc -form cint ${stoptime}-${starttime}` seconds.; echo; echo;")

#Write the preproc script and execute it
ofh = open('_spp_%s.sh' % setname ,'w')
#print "\n".join(sl)+"\n" #DEBUG
ofh.write("\n".join(sl)+"\n")
ofh.close()
if not options.exit:
	system('bash _spp_%s.sh' % setname)
