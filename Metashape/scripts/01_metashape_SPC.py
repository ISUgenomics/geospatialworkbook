################################################################################
# ------ PhotoScan workflow Part 1: --------------------------------------------
# ------ * Image Quality analysis, ---------------------------------------------
# ------ * Camera Alignment analysis, ------------------------------------------
# ------ * Reprojection Error Analysis, ----------------------------------------
# ------ * Sparse Point Cloud Creation, ----------------------------------------
# ------ * Reference settings --------------------------------------------------
################################################################################

# IMPORTS #
import os
import sys
import math
import Metashape as MS
from datetime import datetime

#MS.app.console_pane.clear()                 # comment out when using ISCA

# USER-CUSTOMIZABLE GLOBAL VARIABLES
config_file = "config_file.csv"              # config file with variable-value pairs; must be in the same folder as this script
logfile = open("log.txt", "a")               # file to which verbose mode notes are saved [INFO, WARNING, ERROR]

# AUTOMATIC GLOABL VARIABLES
path = os.getcwd()                           # derive the path of the current directory
params = ["workdir", "doc_title", "data_dir", "coord_system", "spc_quality",
          "marker_coord_file", "marker_coord_system", "export_dir",
          "est_img_quality", "img_quality_cutoff", "reprojection_error",
          "rolling_shutter", "revise_altitude", "altitude_adjust", "depth_filter",
          "marker_type", "marker_tolerance", "marker_min_size", "marker_max_res"]


# MAIN FUNCTION manages calls to subsequent subprocesses:
#   STEP 0: loading config file
#   STEP 1: setting up Metashape application
#   STEP 2: loading & inspecting photos
#   STEP 3: preprocessing of the images
#   STEP 4: building sparse points cloud (SPC)
#   STEP 5: filtering reprojection errors
#   STEP 6: getting number of NOT aligned cameras
#   STEP 7: setting up reference settings
#   STEP 8: detecting markers
def script_setup():
    startTime = datetime.now()              # calculation start time
    logfile.write("Script start time: " + str(startTime) + "\n")
    is_license = True

# - STEP 0: Loading config_file and preparing list of photo inputs
    print("\nSTEP 0: Loading config_file and preparing list of photo inputs...")
    logfile.write("\nSTEP 0: Load config_file and prepare list of photo inputs\n")
    try:
        config, photos = load_config_file(path, config_file)
        name = "/" + config['doc_title'] + ".psx"
    except:
        print("ERROR: STEP 0 FAILED!\n-- Please check the " + logfile + " for the details of the error.")
        logfile.write("EXIT: STEP 0 HAS FAILED!\n")
        sys.exit(1)


# - STEP 1: Seting up Metashape application: doc, chunk
    print("\nSTEP 1: Seting up Metashape application: doc, chunk...")
    logfile.write("\nSTEP 1: Set up Metashape application: doc, chunk\n")
    try:
        doc = MS.app.document                                       # create Metashape (MS) application object
        ## Provide CPU/GPU settings
        MS.app.gpu_mask = 2 ** len(MS.app.enumGPUDevices()) - 1     # activate all available GPUs
        if MS.app.gpu_mask <= 1:                                    # (faster with 1 no difference with 0 GPUs)
            MS.app.cpu_enable = True                                # enable CPU for GPU accelerated processing
        elif MS.app.gpu_mask > 1:                                   # (faster when multiple GPUs are present)
            MS.app.cpu_enable = False                               # disable CPU for GPU accelerated tasks
# ----- WARNING: you need to delete the original automatically created chunk when Metashape opens if running the script from tools
        chunk = MS.app.document.addChunk()                          # create Metashape (MS) chunk
    except:
        print("ERROR: STEP 1 HAS FAILED!\n-- Please check the " + logfile + " for the details of the error.")
        logfile.write("ERROR: Metashape could NOT create a doc or chunk.\nEXIT: STEP 1 HAS FAILED!\n")
        sys.exit(1)


# - STEP 2: Loading & Inspecting photos
    print("\nSTEP 2: Loading & Inspecting photos...")
    logfile.write("\nSTEP 2: Load & Inspect photos\n")
    try:
        load_photos(chunk, config, photos)
    except:
        print("ERROR: STEP 2 HAS FAILED! Loading photos has failed.")
        logfile.write("EXIT: STEP 2 HAS FAILED!\n")
        sys.exit(1)

    if is_license:
        if os.path.isdir(config['workdir']):
            doc.save(config['workdir'] + name)
        else:
            print("ERROR: The path to the working directory: " + config['workdir'] + " does NOT exist. " +
                  "The doc can NOT be saved. Please provide the correct path.")
            logfile.write("ERROR: STEP 2 HAS FAILED!\n--The doc could NOT be saved." +
                  "-- Please check that the path you want to save the file to is correct: " + config['workdir'] + name + "\n")
            sys.exit(1)
    else:
        is_license = False
        print("WARNING: The Metashape license is NOT available. The script will continue the calculation, " +
              "but the doc results will NOT be saved to file. If you want to STOP the calculation now, press CTRL + C on your keyboard.")
        logfile.write("WARNING: The Metashape license is NOT available. Please check if there is a problem with the license server.\n" +
              "The results of your Metashape analysis are NOT saved to a file: " + config['workdir'] + name + "\n")


# - STEP 3: Preprocess images
    print("\nSTEP 3: Preprocessing images results...")
    logfile.write("\nSTEP 3: Preprocess images results\n")
    try:
        orig_n_cams, n_filter_removed, perc_filter_removed, real_qual_thresh = preprocess(config['est_img_quality'], float(config['img_quality_cutoff']), chunk)
    except:
        print("ERROR: STEP 3 HAS FAILED!\n-- Please check the " + logfile + " for the details of the error.")
        logfile.write("EXIT: STEP 3 HAS FAILED!\n")
        sys.exit(1)
    if is_license:
        doc.save(config['workdir'] + name)


# - STEP 4: Build Sparse Point Cloud (SPC)
    print("\nSTEP 4: Building Sparse Point Cloud...")
    logfile.write("\nSTEP 4: Build Sparse Point Cloud\n")
    try:
        points, projections = build_SPC(chunk, config['spc_quality'])
    except:
        print("ERROR: STEP 4 HAS FAILED!\n-- Please check the " + logfile + " for the details of the error.")
        logfile.write("EXIT: STEP 4 HAS FAILED!\n")
        sys.exit(1)
    if is_license:
        doc.save(config['workdir'] + name)


# - STEP 5: Filter reprojection errors
    print("\nSTEP 5: Filtering reprojection errors...")
    logfile.write("\nSTEP 5: Filter reprojection errors\n")
    try:
        total_points, perc_ab_thresh, nselected = filter_reproj_err(chunk, config['reprojection_error'])
    except:
        print("ERROR: STEP 5 HAS FAILED!\n-- Please check the " + logfile + " for the details of the error.")
        logfile.write("EXIT: STEP 5 HAS FAILED!\n")
        sys.exit(1)
    if is_license:
        doc.save(config['workdir'] + name)


# - STEP 6: Get number of NOT aligned cameras
    print("\nSTEP 6: Getting number of NOT aligned cameras...")
    logfile.write("\nSTEP 6: Get number of NOT aligned cameras\n")
    try:
        n_aligned, n_not_aligned = count_aligned(chunk)
    except:
        print("ERROR: STEP 6 HAS FAILED!\n-- Please check the " + logfile + " for the details of the error.")
        logfile.write("EXIT: STEP 6 HAS FAILED!\n")
        sys.exit(1)
    if is_license:
        doc.save(config['workdir'] + name)


# - STEP 7: Set up reference settings & Export settings
    print("\nSTEP 7: Setting up reference settings & Exporting settings...")
    logfile.write("\nSTEP 7: Set up reference settings & Export settings\n")
    try:
        ref_setting_setup(chunk, points, projections)
        export_settings(orig_n_cams, n_filter_removed, perc_filter_removed,
                        real_qual_thresh, n_not_aligned, total_points,
                        nselected, config['workdir'], config['doc_title'])
    except:
        print("ERROR: STEP 7 HAS FAILED!\n-- Please check the " + logfile + " for the details of the error.")
        logfile.write("EXIT: STEP 7 HAS FAILED!\n")
        sys.exit(1)
    if is_license:
        doc.save(config['workdir'] + name)


# - STEP 8: Detect Markers
    print("\nSTEP 8: Detect Markers...")
    logfile.write("\nSTEP 8: Detect Markers\n")
    try:
        detect_markers(chunk, config)
    except:
        print("ERROR: STEP 8 HAS FAILED!\n-- Please check the " + logfile + " for the details of the error.")
        logfile.write("ERROR: There was a problem with marker detection.\nEXIT: STEP 8 HAS FAILED!\n")
        sys.exit(1)
    if is_license:
        doc.save(config['workdir'] + name)


# - Get Execution Time...
    logfile.write("\nTotal Execution Time: " + str(datetime.now() - startTime) + "\n")
    logfile.write("\nPART 1 of the PhotoScan workflow completed successfully!\n")
    logfile.close()
    print("\nPART 1 of the PhotoScan workflow completed successfully!")
    print("\nNow it's time to do some manual cleaning as per the protocol.")


#----------------- Section of functions defining subprocesses -----------------#

# FUNCTION for STEP 0 - LOAD CONFIG FILE & PREPARE LIST OF PHOTOS
def load_config_file(path, config_file):
    config = {}
    # Load 'variable':'value' pairs from the input config_file into the config dictionary
    try:
        config_path = str(path + "/" + config_file).replace('//', '/')
        with open(config_path, 'r') as f:
            for row in f:
                tokens = row.split(',')[:2]
                if tokens[0].strip() in params:
                    config[tokens[0].strip()] = tokens[1].strip()
        if len(config) == len(params):
            logfile.write("INFO: The loaded config variables include: \n")
            for i in config:
                logfile.write("     - " + i + " : " + config[i] + " \n")
        else:
            for i in config.keys():
                if not i in params:
                    logfile.write("ERROR: Your config file does NOT include the required variable: " + i +
                                  " . Please check the spelling carefully.\n")
                    sys.exit(1)
    except Exception:
        logfile.write("ERROR: The " + config_file + " does NOT exist on the " + path + " path.\n")
        logfile.write("       Please copy the required configuration file to the " + path + " directory.\n")
        print("\nERROR in the STEP 0, when loading the variables from the: " + config_file)

    # Locate photos and prepare list of photo inputs
    datadir = config['data_dir']
    if os.path.isdir(datadir) == False:
        datadir = str(config['workdir'] + "/" + datadir).replace('//', '/')
    try:
        os.path.isdir(datadir)
        photos = os.listdir(datadir)                            # get the list of photos filenames
        photos = [os.path.join(datadir, p) for p in photos]     # convert names in the list to full paths
        logfile.write("INFO: " + str(len(photos)) + " photos were found on the path: " + datadir + "\n")
    except Exception:
        logfile.write("ERROR: The following path to folder with photos does NOT exist: " + datadir + "\n")
        print("ERROR in the STEP 0, when preparing the list of photos. The following path to folder with photos does NOT exist: " + datadir)

    return config, photos


# FUNCTION for STEP 2 - LOAD & INSPECT PHOTOS
def load_photos(chunk, config, photos):
    # 1: Add photos to chunk
    try:
        chunk.addPhotos(photos)
        print("-- Photos added successfully.")
    except Exception:
        logfile.write("ERROR: The Metashape chunk.addPhotos() function has failed.\n")

    # 2: Enable rolling shutter compensation
    if config['rolling_shutter'] == 'TRUE':
        chunk.sensors[0].rolling_shutter = True
        print("-- Enabled rolling shutter compenstation.")

    # 3: Define desired Coordinate System and try to set the coordinates for cameras
    new_crs = MS.CoordinateSystem(config['coord_system'])
    try:
        for camera in chunk.cameras:
            camera.reference.location = MS.CoordinateSystem.transform(camera.reference.location,chunk.crs, new_crs)
        print("-- Defined coordinate system.")
    except Exception:
        logfile.write("WARNING: Images do not have projection data... No Worries! It will continue without!")

    # 4: Correct the DJI absolute altitude problems
    # -- WARNING: This portion of the script needs to be checked to see how it interacts with non DJI drone data
    print("-- Trying to correct the DJI absolute altitude problems...")
    for camera in chunk.cameras:
        if not camera.reference.location:
            continue
        elif "DJI/RelativeAltitude" in camera.photo.meta.keys() and config['revise_altitude'] == "TRUE":
            z = float(camera.photo.meta["DJI/RelativeAltitude"]) + float(config['altitude_adjust'])
            camera.reference.location = (camera.reference.location.x, camera.reference.location.y, z)
    print("   DJI corrected successfully.")

    # 5: [OPTIONAL] Import of markers; if a path is given markers are added; if no markers are given then pass
    print("-- Trying to import markers...")
    marker_coord = config['marker_coord_file']
    if marker_coord != "NONE" and os.path.isfile(marker_coord) == False:
        marker_coord = str(config['data_dir'] + "/" + config['marker_coord_file']).replace('//', '/')
    try:
        os.path.isfile(marker_coord)
        config['marker_coord_file'] = marker_coord
        chunk.importReference(marker_coord, columns="nxyzXYZ", delimiter=",",
                              group_delimiters=False, skip_rows=1,
                              ignore_labels=False, create_markers=True, threshold=0.1)
        logfile.write("INFO: Import of the reference for markers was successful.\n")
        print("   Markers added successfully.")

        if config['marker_coord_system'].strip() != config['coord_system'].strip():     # if marker and project crs match then pass otherwise convert marker crs
            for marker in chunk.markers:                                                # this needs testing  but should theoretically work...
                marker.reference.location = new_crs.project(chunk.crs.unproject(marker.reference.location))
            logfile.write("INFO: Import of the marker.reference.location was successful.\n")
    except:
        logfile.write("WARNING: Import of the reference for markers has failed... No Worries! It will continue without!\n")

    # 6: Set project coordinate system
    chunk.crs = new_crs
    chunk.updateTransform
    print("-- Project coordinate system set successfully.")


# FUNCTION for STEP 3 - PREPROCESS IMAGES
def preprocess(est_img_qual, img_qual_thresh, chunk):

    # Estimating Image Quality and excluding poor images
    if est_img_qual == "TRUE":
        print("-- Running image quality filter...")
        chunk.analyzePhotos()  # MSCHANGE

        kept_img_quals = []
        dumped_img_quals = []

        for camera in chunk.cameras:
            IQ = float(camera.meta["Image/Quality"])
            if IQ < float(img_qual_thresh):
                camera.enabled = False
                dumped_img_quals.append(IQ)
            else:
                kept_img_quals.append(IQ)

        n_filter_removed = len(dumped_img_quals)
        orig_n_cams = len(kept_img_quals) + n_filter_removed
        perc_filter_removed = round(((n_filter_removed/orig_n_cams)*100), 1)
        real_qual_thresh = min(kept_img_quals)

        logfile.write("- number of cameras disabled = " + str(n_filter_removed) + "\n")
        logfile.write("- percent of cameras disabled = " + str(perc_filter_removed) + "%\n")
        logfile.write("- number of cameras enabled = " + str(len(kept_img_quals)) + "\n\n")

        for cutoff in [0.9, 0.8, 0.7, 0.6, 0.5]:
            length = len([i for i in kept_img_quals if i >= cutoff])
            logfile.write("-- number of photos with image quality >= " +
                    str(cutoff)+ " : " + str(length) + "\n")
            if length > 0:
                logfile.write("-- percent of cameras with image quality >= " +
                        str(cutoff)+ " : " + str(round((length/orig_n_cams)*100, 1)) + "%\n")
    else:
        logfile.write("Preprocess images skipped.\n")
        print("-- Image quality filtering skipped.")
        chunk.estimateImageQuality()
        all_imgs_qual = []
        for camera in chunk.cameras:
            IQ = float(camera.meta["Image/Quality"])
            all_imgs_qual.append(IQ)

        orig_n_cams = len(chunk.cameras)
        n_filter_removed = "no_filter_applied"
        perc_filter_removed = "no_filter_applied"
        real_qual_thresh = min(all_imgs_qual)

    return orig_n_cams, n_filter_removed, perc_filter_removed, real_qual_thresh


# FUNCTION for STEP 4 - BUILD SPARSE POINTS CLOUD
def build_SPC(chunk, spc_quality):
    print("STEP 4: Building Sparse Point Cloud...")
    scale = {"LowestAccuracy" : 5, "LowAccuracy" : 4, "MediumAccuracy" : 3,
             "HighAccuracy" : 2, "HighestAccuracy" : 1}

    # Match Photos
    # WARNING: Accuracy changed to downscale in Metashape 1.6.4 removed preselection MSCHANGE
    if spc_quality in scale.keys():
        chunk.matchPhotos(downscale=scale[spc_quality], generic_preselection=True,
                          reference_preselection=True, filter_mask=False,
                          keypoint_limit=40000, tiepoint_limit=8000)
        logfile.write("Photos matched with customized SPC quality: " + spc_quality + ".\n")
    else:
        print("WARNING: The entered value for variable: spc_quality is invalid. The default setting will be used: HighestAccuracy.")
        chunk.matchPhotos(downscale=1, generic_preselection=True,
                          reference_preselection=True, filter_mask=False,
                          keypoint_limit=40000, tiepoint_limit=8000)
        logfile.write("Photos matched with default SPC quality: HighestAccuracy.\n")

    # Align Cameras
    try:
        chunk.alignCameras(adaptive_fitting=False)
        point_cloud = chunk.point_cloud
        points = point_cloud.points
        projections = chunk.point_cloud.projections
        logfile.write("Cameras aligned successfully.\n")
    except:
        logfile.write("ERROR: There was a problem while aligning the cameras and/or creating projections.\n")
        print("ERROR in the STEP 4, when aligning cameras.")

    return points, projections


# FUNCTION for STEP 5 - FILTER REPROJECTION ERRORS
# Filter points by their reprojection error and remove those with values > 0.45 (or the limit set in the input file)
def filter_reproj_err (chunk, reproj_err_limit):

    logfile.write("Filtering tiepoints by reprojection error (threshold = " +
            str(reproj_err_limit) + ").\n")
    reproj_err_limit = float(reproj_err_limit)
    f = MS.PointCloud.Filter()
    f.init(chunk, MS.PointCloud.Filter.ReprojectionError)
    f.selectPoints(reproj_err_limit)
    nselected = len([p for p in chunk.point_cloud.points if p.selected])
    total_points = len(chunk.point_cloud.points)
    perc_ab_thresh = round((nselected/total_points*100), 1)

    if perc_ab_thresh > 20:
        print ("---------------------------------------------------------")
        print ("WARNING >20% OF POINTS ABOVE REPROJECTION ERROR THRESHOLD")
        print ("---------------------------------------------------------")
        logfile.write("WARNING: more than 20% of points above reprojection error threshold!\n")

    logfile.write("Number of points below threshold: " + str(reproj_err_limit) +
          "\nReprojection Error Limit: " + str(nselected) + "/" +
          str(total_points) + " (" + str(perc_ab_thresh) + "%)\n")

    print("Removing points above error threshold...")
    f.removePoints(reproj_err_limit)
    logfile.write("Removed points above error threshold.\n")

    return total_points, perc_ab_thresh, nselected


# FUNCTION for STEP 6 - COUNT NOT ALIGNED CAMERAS
def count_aligned(chunk):
    aligned_list = []
    for camera in chunk.cameras:
        if camera.transform:
            aligned_list.append(camera)

    not_aligned_list = []
    for camera in chunk.cameras:
        if not camera.transform:
            not_aligned_list.append(camera)

    n_aligned = len(aligned_list)
    n_not_aligned = len(not_aligned_list)
    sum = n_aligned + n_not_aligned
    try:
        val = n_aligned / sum * 100
        logfile.write("Number (%) of aligned cameras is: " + str(n_aligned) + " (" + str(val)+ "%)\n")
    except ZeroDivisionError:
        logfile.write("WARNING: No cameras are aligned!")
    try:
        val = n_not_aligned/sum*100
        logfile.write("Number of cameras NOT aligned is: " + str(n_not_aligned) + " (" + str(val)+ "%)\n")
    except ZeroDivisionError:
        logfile.write("WARNING: No cameras loaded - something isn't aligned...\n")

    return (n_aligned, n_not_aligned)


# FUNCTION for STEP 7A - SET UP REFERENCE SETTINGS
def ref_setting_setup(chunk, points, projections):
    cam_loc_acc = [15, 15, 20]                                          # xyz metres
    mark_loc_acc = [0.02, 0.02, 0.05]                                   # xyz metres
    mark_proj_acc = 2                                                   # pixels

    chunk.camera_location_accuracy = cam_loc_acc                        # units in m
    chunk.marker_location_accuracy = mark_loc_acc                       # SINGLE VALUE USED WHEN MARKER-SPECIFIC ERRORS ARE UNAVAILABLE
    chunk.marker_projection_accuracy = mark_proj_acc                    # FOR MANUALLY PLACED MARKERS

    total_error = calc_reprojection_error(chunk, points, projections)   # calculate reprojection error
    reproj_error = sum(total_error)/len(total_error)                    # get average rmse for all cameras

    logfile.write("Mean reprojection error for point cloud: " + str(round(reproj_error, 3)) + "\n")
    logfile.write("Max reprojection error is: " + str(round(max(total_error), 3)) + "\n")

    if reproj_error < 1:
        reproj_error = 1.0
        chunk.tiepoint_accuracy = round(reproj_error, 2)


# INTERNAL FUNCTION for STEP 7A - CALC REPROJECTION ERROR
def calc_reprojection_error(chunk, points, projections):
    npoints = len(points)
    photo_avg = []

    for camera in chunk.cameras:
        if not camera.transform:
            continue
        point_index = 0
        photo_num = 0
        photo_err = 0
        for proj in projections[camera]:
            track_id = proj.track_id
            while point_index < npoints and points[point_index].track_id < track_id:
                point_index += 1
            if point_index < npoints and points[point_index].track_id == track_id:
                if not points[point_index].valid:
                    continue
                dist = camera.error(points[point_index].coord, proj.coord).norm() ** 2      # get the square error for each point in camera
                photo_num += 1                                                              # counts number of points per camera
                photo_err += dist                                                           # creates list of square point errors
        photo_avg.append(math.sqrt(photo_err / photo_num))                                  # get root mean square error for each camera

    return photo_avg                                                                        # returns list of rmse values for each camera


# FUNCTION for STEP 7B - EXPORT SETTINGS
def export_settings(orig_n_cams, n_filter_removed, perc_filter_removed,
                    real_qual_thresh, n_not_aligned, total_points,
                    nselected, workdir, doc_title):
    print("Exporting settings to temp_folder...")

    opt_list = {"n_cameras_loaded" : orig_n_cams,
                "n_cams_removed_qual_filter" : n_filter_removed,
                "%_cams_removed_qual_filter" : perc_filter_removed,
                "img_qual_min_val" : real_qual_thresh,
                "n_cameras_not_aligned" : n_not_aligned,
                "n_points_orig_SPC" : total_points,
                "n_points_removed_reproj_filter" : nselected
               }

    try:
        tmp_path = str(workdir + '/' + doc_title + '.files').replace ('//', '/')
        if not os.path.isdir(tmp_path):
            os.mkdir(tmp_path)
        export_file = open(tmp_path + "/PhSc1_exported_settings.csv", "a")
        export_file.write("variable,value\n")
        for i in opt_list.keys():
            export_file.write(i + "," + str(opt_list[i]) + "\n")
        export_file.close()

    except:
        print("\nWARNING: The settings can NOT be exported to the temporary directory: " + tmp_path)
        logfile.write("WARNING: The settings can NOT be exported to the temporary directory: " + tmp_path + " .\n")


# FUNCTION for STEP 8 - DETECT MARKERS
def detect_markers(chunk, config):
    # Detect Markers
    try:
        chunk.detectMarkers(target_type = getattr(Metashape.TargetType, config['marker_type']),
                            tolerance = config['marker_tolerance'],
                            minimum_size = config['marker_min_size'],
                            maximum_residual = config['marker_max_res'])
        logfile.write(str(len(chunk.markers)) + " markers were detected.\n")
    except:
        print("ERROR in STEP 8, when detecting markers.")
        sys.exit(1)

    # Read in the measured GCPs.
    if len(chunk.markers) > 0:
        try:
            marker_coords = config['marker_coord_file']
            os.path.isfile(marker_coords)
        except:
            print("ERROR in STEP 8, when reading the measured GCPs.\n")
            logfile.write("ERROR in STEP 8, when reading the measured GCPs.\n" + "      The provided file is: " + marker_coords +
                          "\n      Please make sure you have entered the correct data file name and its content is not empty.")
        gcplist = []
        with open(marker_coords, 'r') as f:
            for row in f:
                gcpline = row.split(',')
                if len(gcpline) == 4 and gcpline[0].isnumeric():
                    try:
                        gcplist.append([float(gcpline[1]), float(gcpline[2]), float(gcpline[3])])
                    except:
                        print("WARNING: Skipped a line that might have been a header line or an invalid GCP entry: " + str(row))
        logfile.write(str(len(gcplist)) + " GCP entries were collected.\n")

        # See: https://github.com/campbell-ja/MetashapePythonScripts/blob/main/_Workflow-IntheRound/Metashape_PythonScripts_04-Part_02_DetectMarkers-OptimizeMarkerError.py
        # Remove Markers with less than 9 number of projections
        if len(gcplist):
            for marker in chunk.markers:
                if len(marker.projections) < 9:
                    chunk.remove(marker)
                    logfile.write("Removed marker: " + marker.label + " for having only " + str(len(marker.projections)) + " projections.\n")
                elif len(gcplist) > 0:
                    # Now let's compare our detected markers to our measured points.
                    # Need to convert from pixel coordinates to projection.
                    T = chunk.transform.matrix
                    pos = chunk.crs.project(T.mulp(marker.position))
                    # Look through GCPs...
                    for gcp in gcplist:
                        dist = math.sqrt((gcp[0] - pos.x)**2 + (gcp[1] - pos.y)**2)
                        # If our distance is within... three meters? ...assign this GCPs x/y/z values.
                        if dist < 3.0:
                            marker.reference.location = MS.Vector([gcp[0], gcp[1], gcp[2]])
                            logfile.write("Changed reference location for marker: " + marker.label + ", " + str(marker.reference.location) + "\n")
                            break
            logfile.write("After this step " + str(len(chunk.markers)) + " markers were kept.\n")


# START EXECUTING FROM THE MAIN FUNCTION
if __name__ == '__main__':
    script_setup()
