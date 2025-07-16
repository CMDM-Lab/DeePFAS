### Install Bioconductor package
if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install()

### Install XCMS package
#BiocManager::install("xcms")
### Install MSnbase package
BiocManager::install("MSnbase")
### Install IPO package
BiocManager::install("IPO")
## Install Spectra package
BiocManager::install("Spectra")
BiocManager::install("MsBackendMsp")
BiocManager::install("MetaboAnnotation")

### Import library
.libPaths(c("/usr/local/lib/R/site-library", "/usr/lib/R/site-library", "/usr/lib/R/library"))
library(xcms)
library(IPO)
library(Spectra)
library('MSnbase')
library(MsBackendMsp)
library(MetaboAnnotation)

input.arg <- commandArgs(TRUE)

### Import Data
exp_path <- input.arg[1]
#input file format: mzXML
input_file <- input.arg[2]
#ionization mode: positive or negative mode
if (input.arg[3] == "positive") {
  polarity <- "pos"
}else if (input.arg[3] == "negative") {
  polarity <- "neg"
}else {
  polarity <- "neg"
}
raw_data <- readMSData(input_file, mode = "onDisk")
nist_hr_msp_path <- paste0("/usr/local/app/lib/peak_detection/", "hr_msms_nist_", polarity, ".MSP")
nist_lr_msp_path <- paste0("/usr/local/app/lib/peak_detection/", "lr_msms_nist_", polarity, ".MSP")
nist_hr_lib_path <- paste0("/usr/local/app/lib/peak_detection/", "nist_hr_lib_", polarity, "_r.var")
nist_lr_lib_path <- paste0("/usr/local/app/lib/peak_detection/", "nist_lr_lib_", polarity, "_r.var")

best_parameters <- paste0(exp_path, '/best_parameters.var')
peak_list_var <- paste0(exp_path, '/peak_list.var')
peak_table <- paste0(exp_path, '/peak_list.csv')
peak_spectra <- paste0(exp_path, '/peak_spectra.var')
peak_spectra_msp <- paste0(exp_path, '/peak_spectra.msp')
peak_matched_hr <- paste0(exp_path, '/peak_matched_hr.var')
peak_matched_lr <- paste0(exp_path, '/peak_matched_lr.var')
peak_annotation_hr_file <- paste0(exp_path, '/peak_annotation_hr_result.csv')
peak_annotation_lr_file <- paste0(exp_path, '/peak_annotation_lr_result.csv')

### IPO(parameter optimization)
if (file.exists(best_parameters)) {
  print("best parameters file existed!")
  load(best_parameters)
} else {
  peakpickingParameters <- getDefaultXcmsSetStartingParams('centWave')
  peakpickingParameters$min_peakwidth <- c(16, 24)
  peakpickingParameters$max_peakwidth <- c(36, 60)
  peakpickingParameters$ppm <- c(10, 30)
  peakpickingParameters$mzdiff <- c(0.001, 0.005)
  time.xcmsSet <- system.time({ # measuring time
    resultPeakpicking <-
      optimizeXcmsSet(files = input_file,
                      params = peakpickingParameters,
                      nSlaves = 1,
                      subdir = NULL,
                      plot = F)
  })
  save(resultPeakpicking, file=best_parameters)
}

best_min_peakwidth <- resultPeakpicking$best_settings$parameters$min_peakwidth
best_max_peakwidth <- resultPeakpicking$best_settings$parameters$max_peakwidth
best_ppm <- resultPeakpicking$best_settings$parameters$ppm
best_mzdiff <- resultPeakpicking$best_settings$parameters$mzdiff
best_snthresh <- resultPeakpicking$best_settings$parameters$snthresh

### Find chrompeaks with optimized parameters
if (file.exists(peak_table)) {
  print("peak table existed!")
  load(peak_list_var)
} else {
  cwp <- CentWaveParam(peakwidth = c(best_min_peakwidth, best_max_peakwidth),
                       ppm = best_ppm,
                       snthresh = best_snthresh,
                       mzdiff = best_mzdiff)
  cp_data <- findChromPeaks(raw_data, param = cwp)
  save(cp_data, file=peak_list_var)

  ### Export peak_list
  result <- chromPeaks(cp_data)
  write.csv(result, file=peak_table)
}
### Export spectra(.msp)
if (file.exists(peak_spectra)) {
  print("peak spectra file existed!")
  load(peak_spectra)
} else {
  peak_spectra_data <- chromPeakSpectra(cp_data, msLevel = 2L, return.type = "Spectra")
  #fl <- tempfile()
  export(peak_spectra_data, backend = MsBackendMsp(),
         file=peak_spectra_msp,
         mapping = spectraVariableMapping(MsBackendMsp())
         )
  save(peak_spectra_data, file=peak_spectra)
}

### Spectra match
if (file.exists(peak_matched_hr)) {
  print("matched peak file existed!")
} else {
  # --- Low-Resolution Spectra Matching ---
  if(file.exists(nist_lr_lib_path)){
    print("Loading NIST LR MSMS Lib")
    load(nist_lr_lib_path)
  }else{
    nist_lr_library <- Spectra(readMsp(nist_lr_msp_path, msLevel = 2L, mapping = spectraVariableMapping(MsBackendMsp())))
    save(nist_lr_library, file=nist_lr_lib_path)
  }

  prm <- CompareSpectraParam(
    ppm = 10,
    requirePrecursor = FALSE,
    THRESHFUN = function(x) which(x >= 0.7)
  )
  mtch_lr <- matchSpectra(peak_spectra_data, nist_lr_library, param = prm)
  save(mtch_lr, file=peak_matched_lr)
  #pandoc.table(style = "rmarkdown", as.data.frame(spectraData(mtch_lr, c("rtime", "target_compound_name", "score"))))
  peak_annotation_lr_result <- spectraData(mtch_lr, c("rtime", "target_Name", "target_NISTNO", "target_CASNO", "score"))
  write.csv(peak_annotation_lr_result, file=peak_annotation_lr_file)
  rm(nist_lr_library) # Remove the low-res library to free memory

  # --- High-Resolution Spectra Matching (Uncommented and Annotated) ---
  # Check if the pre-processed high-resolution NIST library exists.
  if(file.exists(nist_hr_lib_path)){
    print("Loading NIST HR MSMS Lib")
    load(nist_hr_lib_path) # Load the pre-processed high-res library.
  }else{
    # If not, read the high-res MSP file and convert it into a 'Spectra' object.
    nist_hr_library <- Spectra(readMsp(nist_hr_msp_path, msLevel = 2L, mapping = spectraVariableMapping(MsBackendMsp())))
    save(nist_hr_library, file=nist_hr_lib_path) # Save the processed high-res library for future use.
  }
  # Record the start time for performance measurement.
  start_time <- Sys.time()
  # Perform spectral matching between the extracted peak spectra and the NIST high-resolution library.
  # The same 'prm' (CompareSpectraParam) is used as for low-resolution matching.
  mtch_hr <- matchSpectra(peak_spectra_data, nist_hr_library, param = prm)
  # Record the end time.
  end_time <- Sys.time()
  # Print the time taken for high-resolution matching.
  print(end_time - start_time)
  # Save the results of the high-resolution spectral matching.
  save(mtch_hr, file=peak_matched_hr)
  # Extracts relevant data (retention time, target name, NIST number, CAS number, score)
  # from the high-resolution matching results for annotation.
  peak_annotation_hr_result <- spectraData(mtch_hr, c("rtime", "target_Name", "target_NISTNO", "target_CASNO", "score"))
  # Writes the high-resolution annotation results to a CSV file.
  write.csv(peak_annotation_hr_result, file=peak_annotation_hr_file)
  rm(nist_hr_library) # Remove the high-res library from memory to free up space.
}
