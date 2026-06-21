import cv2
import numpy as np
import glob
import sys

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics library not found. Please, run 'pip install ultralytics' to install it.")
    sys.exit(1)


# Method to obtain the bird-eye view and the transformation matrix
def get_birds_eye_view(image):
    f_x, f_y = 1970.0, 1970.0
    c_x, c_y = 970.0, 483.0
    h = 1.6600 
    min_distance, max_distance = 5.5, 25.0
    lane_weight = 2.5
    
    p3d = np.float32([
        [-lane_weight, h, max_distance], [lane_weight, h, max_distance],
        [lane_weight, h, min_distance], [-lane_weight, h, min_distance]
    ])
    def project(p):
        return [np.float32((f_x * p[0] / p[2]) + c_x), np.float32((f_y * p[1] / p[2]) + c_y)]

    src_pts = np.array([project(p) for p in p3d], dtype=np.float32)
    px_m = 50
    w = int((lane_weight * 2) * px_m)
    h_out = int((max_distance - min_distance) * px_m)
    dst_pts = np.float32([[0, 0], [w - 1, 0], [w - 1, h_out - 1], [0, h_out - 1]])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    Minv = cv2.getPerspectiveTransform(dst_pts, src_pts)
    
    return cv2.warpPerspective(image, M, (w, h_out)), M, Minv, px_m, max_distance


# Lane Enhancement with the two kernels
def apply_lane_enhancement(gray_img):
    sigma_x = 1.5
    sigma_y = 3.0
    ksize_x = int(6 * sigma_x + 1) | 1 
    ksize_y = int(6 * sigma_y + 1) | 1

    x = np.linspace(-(ksize_x // 2), ksize_x // 2, ksize_x)
    gx = (1 / sigma_x**2) * np.exp(-(x**2) / (2 * sigma_x**2)) * (1 - (x**2 / sigma_x**2))
    
    y = np.linspace(-(ksize_y // 2), ksize_y // 2, ksize_y)
    gy = np.exp(-(y**2) / (2 * sigma_y**2))

    enhanced = cv2.sepFilter2D(gray_img.astype(np.float32), -1, gx, gy)
    enhanced[enhanced < 0] = 0
    
    if np.max(enhanced) > 0:
        enhanced = (enhanced / np.max(enhanced)) * 255.0
        
    return enhanced.astype(np.uint8)


# Image binarization
def apply_iterative_threshold(gray_img):
    g_min, g_max = float(np.min(gray_img)), float(np.max(gray_img))
    th = (g_max + g_min) / 2.0
    while True:
        region_A = gray_img[gray_img >= th]
        region_B = gray_img[gray_img < th]
        g_A = np.mean(region_A) if len(region_A) > 0 else 0
        g_B = np.mean(region_B) if len(region_B) > 0 else 0
        new_th = (g_A + g_B) / 2.0
        if abs(new_th - th) < 0.5: break
        th = new_th
    _, binary = cv2.threshold(gray_img, th, 255, cv2.THRESH_BINARY)
    return binary, th



def draw_dashed_line(img, pt1, pt2, color, thickness=2, dash_length=15):
    dist = np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)
    pts = np.linspace(0, 1, int(dist / dash_length))
    for i in range(len(pts) - 1):
        if i % 2 == 0:
            start = (int(pt1[0] + (pt2[0]-pt1[0])*pts[i]), int(pt1[1] + (pt2[1]-pt1[1])*pts[i]))
            end = (int(pt1[0] + (pt2[0]-pt1[0])*pts[i+1]), int(pt1[1] + (pt2[1]-pt1[1])*pts[i+1]))
            cv2.line(img, start, end, color, thickness)



def main():
    if len(sys.argv) < 2:
        print("Use: python run_gold.py <imgs_path> [debug]")
        sys.exit(1)

    image_paths = sorted(glob.glob(sys.argv[1]))
    if not image_paths:
        print("No images found in the specified path.")
        sys.exit(1)

    # check if debug arg is asserted
    debug_mode = any("debug" in arg.lower() for arg in sys.argv)

    # yolo inizialization
    print("Loading YOLO model...")
    model = YOLO('yolov8n.pt') 
    print("YOLO loaded. Press ESC or 'q' to quit.")

    current_idx = 0
    display_h = 350 if debug_mode else 550 # the window shrinks if needed for debug mode
    is_paused = False # flag to manage the pause, just in debug mode 

    while True:
        original = cv2.imread(image_paths[current_idx])
        if original is None: break
        out_orig = original.copy()
        
        # 1. BEV 
        bev, M, Minv, px_m, dist_max = get_birds_eye_view(original)
        out_bev = bev.copy()
        
        # yolo ---
        results = model(original, verbose=False)
        coco_classes = {0: "Person", 1: "Bicycle", 2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

        blackout_polygons = [] # objects array to be deleted from the binarized img 
        
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls in coco_classes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # distance and bev coordinates
                    bx = (x1 + x2) / 2
                    by = y2
                    
                    # projects the top-left/right corners
                    pts_orig = np.float32([[[bx, by], [x1, by], [x2, by]]])
                    pts_bev = cv2.perspectiveTransform(pts_orig, M)
                    
                    # extracts the x coordinates of the sides and the y of the center of the bb
                    bev_x_left = int(pts_bev[0][1][0])
                    bev_x_right = int(pts_bev[0][2][0])
                    bev_y = int(pts_bev[0][0][1])
                    
                    distanza_m = dist_max - (bev_y / px_m)
                    
                    h_bev, w_bev = bev.shape[:2]
                    
                    is_inside_x = (bev_x_left < w_bev) and (bev_x_right > 0)
                    is_inside_y = (0 <= bev_y < h_bev)
                    
                    if is_inside_x and is_inside_y:
                        # if inside bev: red bb + label with class and distance
                        cv2.rectangle(out_orig, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                        label = f"{coco_classes[cls]} {distanza_m:.1f}m"
                        cv2.putText(out_orig, label, (int(x1) + 8, int(y2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)   

                        if y2 > 485:
                            pad = 15
                            ex1 = x1 - pad
                            ex2 = x2 + pad
                            ey1 = max(485, y1 - pad)
                            ey2 = y2 + pad
                            
                            box_corners = np.float32([[[ex1, ey1], [ex2, ey1], [ex2, ey2], [ex1, ey2]]])
                            bev_box_corners = cv2.perspectiveTransform(box_corners, M)
                            blackout_polygons.append(np.int32(bev_box_corners))
                    else:
                        # if out of the bev the bb is orange
                        cv2.rectangle(out_orig, (int(x1), int(y1)), (int(x2), int(y2)), (0, 165, 255), 2)

        # 1.2 preprocessing
        gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 2. Lane Enhancement
        enhanced_img = apply_lane_enhancement(blurred)
        
        # 3. binarization
        binary, _ = apply_iterative_threshold(enhanced_img)

        # ENHANCEMENT: deleting the bounding boxes of the detected objects from the binary image 
        for poly in blackout_polygons:
            cv2.fillPoly(binary, [poly], 0)
        
        # 4. histogram and Peak Detection
        histogram = np.sum(binary, axis=0)
        mid = binary.shape[1] // 2
        peaks = [np.argmax(histogram[:mid]), np.argmax(histogram[mid:]) + mid]
        
        lanes_found = False
        
        for p_x in peaks:
            if histogram[p_x] > (binary.shape[0] * 255 * 0.10):
                lanes_found = True
                
                col_slice = binary[:, max(0, p_x-5):min(binary.shape[1], p_x+5)]
                white_rows = np.any(col_slice == 255, axis=1)
                vertical_fill_ratio = np.sum(white_rows) / binary.shape[0]
                is_dashed = vertical_fill_ratio < 0.55
                
                pt1_bev, pt2_bev = (p_x, 0), (p_x, bev.shape[0])
                pts_bev = np.float32([[pt1_bev], [pt2_bev]])
                pts_orig = cv2.perspectiveTransform(pts_bev, Minv)
                pt1_orig = tuple(pts_orig[0][0].astype(int))
                pt2_orig = tuple(pts_orig[1][0].astype(int))
                
                color = (0, 255, 0)
                if is_dashed:
                    draw_dashed_line(out_bev, pt1_bev, pt2_bev, color, 3)
                    draw_dashed_line(out_orig, pt1_orig, pt2_orig, color, 3)
                else:
                    cv2.line(out_bev, pt1_bev, pt2_bev, color, 3)
                    cv2.line(out_orig, pt1_orig, pt2_orig, color, 3)



        # UI Layout 
        aspect_orig = original.shape[1] / original.shape[0]
        w_orig = int(display_h * aspect_orig)
        
        aspect_bev = bev.shape[1] / bev.shape[0]
        w_bev = int(display_h * aspect_bev) * 3
        
        hist_canvas = np.zeros((bev.shape[0], bev.shape[1], 3), dtype=np.uint8)
        for i, val in enumerate(histogram):
            cv2.line(hist_canvas, (i, bev.shape[0]), (i, bev.shape[0] - int(val/255)), (255, 0, 255), 1)
        
        res_orig = cv2.resize(out_orig, (w_orig, display_h))
        res_hist = cv2.resize(hist_canvas, (w_orig, display_h)) 
        res_bev = cv2.resize(out_bev, (w_bev, display_h)) 

        # conversion from b&w to rgb to show the debug bounding boxes
        bin_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        # red bounfing boxes
        if debug_mode:
            for poly in blackout_polygons:
                cv2.polylines(bin_bgr, [poly], isClosed=True, color=(0, 0, 255), thickness=2)
                
        res_bin = cv2.resize(bin_bgr, (w_bev, display_h)) 

        if not lanes_found:
            cv2.putText(res_orig, "No lanes found", (res_orig.shape[1]//2 - 150, res_orig.shape[0]//2 + 130), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.putText(res_bev, "No lanes found", (65, res_bev.shape[0]//2 + 130), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        if debug_mode:
            left_col = np.vstack((res_orig, res_hist))
            right_col = np.vstack((res_bev, res_bin))
            grid = np.hstack((left_col, right_col))
        else:
            grid = np.hstack((res_orig, res_bev))

        cv2.imshow("Lane Detection GOLD with YOLO", grid)
        

        # input handling
        key = cv2.waitKey(100) & 0xFF
        
        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            is_paused = not is_paused # Alterna lo stato di pausa
            
        try:
            if cv2.getWindowProperty("Lane Detection GOLD with YOLO", cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break
            
        if debug_mode and is_paused:
            # pause available only if in debug mode
            while is_paused:
                key_pause = cv2.waitKey(100) & 0xFF
                
                if key_pause == ord('q') or key_pause == 27:
                    return
                elif key_pause == ord(' '):
                    is_paused = False 
                    
                try:
                    if cv2.getWindowProperty("Lane Detection GOLD with YOLO", cv2.WND_PROP_VISIBLE) < 1:
                        return
                except cv2.error:
                    return
        
        # frame scrolling logic
        current_idx += 1
        if current_idx >= len(image_paths):
            current_idx = 0

    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()