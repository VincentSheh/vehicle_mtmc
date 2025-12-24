import cv2
import numpy as np

# VIDEO_PATH = "assets/bellevue_150th_eastgate/Bellevue_150th_Eastgate__2017-09-11_07-08-31.mp4"   # change this
VIDEO_PATH = "./Bellevue_150th_Eastgate__2017-09-11_07-08-31.mp4"   # change this
SAVE_MASK_PATH = "./Bellevue_150th_Eastgate__2017-09-11_07-08-31bellevue_150th_eastgate/Bellevue_150th_Eastgate__2017-09-11_07-08-31.png"  # optional

points = []
drawing_done = False


def mouse_callback(event, x, y, flags, param):
    global points, drawing_done

    if drawing_done:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point added: {(x, y)}")


def main():
    global drawing_done

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Failed to read frame")

    clone = frame.copy()

    cv2.namedWindow("Draw ROI (press ENTER to finish, R to reset)")
    cv2.setMouseCallback("Draw ROI (press ENTER to finish, R to reset)", mouse_callback)

    while True:
        display = clone.copy()

        # draw points
        for p in points:
            cv2.circle(display, p, 4, (0, 255, 0), -1)

        # draw polygon lines
        if len(points) >= 2:
            cv2.polylines(
                display,
                [np.array(points, dtype=np.int32)],
                isClosed=False,
                color=(0, 255, 0),
                thickness=2,
            )

        cv2.imshow("Draw ROI (press ENTER to finish, R to reset)", display)
        key = cv2.waitKey(1) & 0xFF

        if key == 13:  # ENTER
            if len(points) >= 3:
                drawing_done = True
                break
            else:
                print("Need at least 3 points to form a polygon")

        elif key == ord("r"):
            points.clear()
            print("Reset points")

        elif key == 27:  # ESC
            cv2.destroyAllWindows()
            return

    # create mask
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)

    # apply mask (keep inside polygon)
    masked = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("Masked Result", masked)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # save mask if needed
    cv2.imwrite(SAVE_MASK_PATH, mask)
    print(f"Mask saved to {SAVE_MASK_PATH}")


if __name__ == "__main__":
    main()