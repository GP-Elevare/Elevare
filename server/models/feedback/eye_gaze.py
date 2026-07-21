def analyze_video_gaze_headless(video_path):
    """
    Analyzes an MP4 video to determine the percentage of time the user is looking 
    directly at the camera. Runs in headless mode (no video window).
    
    Parameters:
        video_path (str): Path to the input MP4 file.
        
    Returns:
        float: Percentage of frames the user was looking at the camera.
    """
    import cv2
    import numpy as np
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    import os

    # Ensure the model file exists before starting
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")


    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing required model file: {model_path}. Please download it and place it in the script directory.")

    # 1. Initialize the modern MediaPipe Face Landmarker
    base_options = python.BaseOptions(model_asset_path=model_path)
    
    # We use VIDEO mode to take advantage of MediaPipe's frame-to-frame tracking
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1
    )

    # MediaPipe Landmark Indices
    LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    LEFT_IRIS_CENTER = 468
    RIGHT_IRIS_CENTER = 473

    # Sensitivity Thresholds (0.5 means perfectly centered iris)
    H_THRESH_LOW, H_THRESH_HIGH = 0.42, 0.58
    V_THRESH_LOW, V_THRESH_HIGH = 0.40, 0.60

    total_frames = 0
    looking_at_camera_frames = 0

    print(f"Processing '{video_path}' in headless mode...")
    cap = cv2.VideoCapture(video_path)
    
    # Get FPS to calculate video timestamps accurately for the new API
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30 # Fallback just in case

    # 2. Open the landmarker in a context manager
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                break  # End of video

            total_frames += 1
            
            # Simple progress feedback in console
            if total_frames % 30 == 0:
                print(f"Processing frame {total_frames}...", end="\r")

            # 3. Convert to MediaPipe Image format
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Calculate timestamp in milliseconds
            timestamp_ms = int((total_frames / fps) * 1000)

            # 4. Process using the Tasks API
            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            # 5. Extract logic (Modern API returns a list of landmarks directly)
            if results.face_landmarks:
                # Get the first face detected
                face_landmarks = results.face_landmarks[0]
                
                # Map landmarks to pixel coordinates (Note: the syntax is slightly different)
                mesh_points = np.array([
                    np.multiply([p.x, p.y], [frame.shape[1], frame.shape[0]]).astype(int) 
                    for p in face_landmarks
                ])
                
                # Extract coordinates
                left_iris = mesh_points[LEFT_IRIS_CENTER]
                right_iris = mesh_points[RIGHT_IRIS_CENTER]
                
                # --- Left Eye Ratio Calculation ---
                left_coords = mesh_points[LEFT_EYE]
                l_min_x, l_max_x = np.min(left_coords[:, 0]), np.max(left_coords[:, 0])
                l_min_y, l_max_y = np.min(left_coords[:, 1]), np.max(left_coords[:, 1])
                l_w, l_h = l_max_x - l_min_x, l_max_y - l_min_y
                
                left_h_ratio = (left_iris[0] - l_min_x) / l_w if l_w > 0 else 0.5
                left_v_ratio = (left_iris[1] - l_min_y) / l_h if l_h > 0 else 0.5

                # --- Right Eye Ratio Calculation ---
                right_coords = mesh_points[RIGHT_EYE]
                r_min_x, r_max_x = np.min(right_coords[:, 0]), np.max(right_coords[:, 0])
                r_min_y, r_max_y = np.min(right_coords[:, 1]), np.max(right_coords[:, 1])
                r_w, r_h = r_max_x - r_min_x, r_max_y - r_min_y
                
                right_h_ratio = (right_iris[0] - r_min_x) / r_w if r_w > 0 else 0.5
                right_v_ratio = (right_iris[1] - r_min_y) / r_h if r_h > 0 else 0.5
                
                # --- Average and Evaluate ---
                avg_h_ratio = (left_h_ratio + right_h_ratio) / 2
                avg_v_ratio = (left_v_ratio + right_v_ratio) / 2

                if (H_THRESH_LOW <= avg_h_ratio <= H_THRESH_HIGH) and (V_THRESH_LOW <= avg_v_ratio <= V_THRESH_HIGH):
                    looking_at_camera_frames += 1

    cap.release()

    if total_frames == 0:
        print("\nError: Could not process any frames.")
        return 0.0

    percentage = (looking_at_camera_frames / total_frames) * 100
    
    print(f"\nAnalysis complete. Attention Score: {percentage:.2f}%")
    return percentage