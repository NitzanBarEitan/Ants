import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for saving figures
import matplotlib.pyplot as plt
import time
from scipy.signal import hilbert, correlate, butter, filtfilt
from scipy.fft import fft, fftfreq

def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    """Apply a Butterworth bandpass filter to isolate the oscillatory component."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def select_rois(cap, scale_factor=0.75):
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Unable to capture initial frame.")
    # If scale_factor is 1, the full resolution is shown.
    if scale_factor != 1:
        frame_display = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
    else:
        frame_display = frame.copy()
    cv2.namedWindow("Initial Frame", cv2.WINDOW_NORMAL)
    cv2.imshow("Initial Frame", frame_display)
    cv2.waitKey(1)
    print("Select ROI for object 1 and press ENTER/SPACE.")
    roi1 = cv2.selectROI("Initial Frame", frame_display, fromCenter=False, showCrosshair=True)
    print("Select ROI for object 2 and press ENTER/SPACE.")
    roi2 = cv2.selectROI("Initial Frame", frame_display, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Initial Frame")
    return roi1, roi2, frame_display

def select_calibration_points(frame):
    points = []
    clone = frame.copy()
    def click_event(event, x, y, flags, param):
        nonlocal points, clone
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(clone, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Calibration", clone)
    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.imshow("Calibration", clone)
    cv2.setMouseCallback("Calibration", click_event)
    print("Select two points on the frame that represent a known real-world distance.")
    while len(points) < 2:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyWindow("Calibration")
    return points

def dynamic_analysis_plots(times_array, x1_vals, x2_vals, fs=30, lowcut=0.5, highcut=5.0):
    # Detrend signals by subtracting the mean
    x1_detrended = x1_vals - np.mean(x1_vals)
    x2_detrended = x2_vals - np.mean(x2_vals)
    
    # Apply bandpass filtering to isolate the oscillatory behavior
    x1_filtered = bandpass_filter(x1_detrended, lowcut, highcut, fs)
    x2_filtered = bandpass_filter(x2_detrended, lowcut, highcut, fs)
    
    # Compute analytic signals and phase using the Hilbert transform on the filtered signals
    analytic_signal1 = hilbert(x1_filtered)
    analytic_signal2 = hilbert(x2_filtered)
    phase1 = np.unwrap(np.angle(analytic_signal1))
    phase2 = np.unwrap(np.angle(analytic_signal2))
    phase_diff = phase1 - phase2

    # Global normalization for correlation analysis
    x1_global = (x1_vals - np.mean(x1_vals)) / np.std(x1_vals)
    x2_global = (x2_vals - np.mean(x2_vals)) / np.std(x2_vals)

    # Sliding-window normalized cross-correlation with global normalization
    window_size = 50  # Adjust based on your data characteristics
    step_size = window_size // 2
    corr_max = []
    times_corr = []
    for i in range(0, len(x1_global) - window_size + 1, step_size):
        win1 = x1_global[i:i+window_size]
        win2 = x2_global[i:i+window_size]
        corr_win = correlate(win1, win2, mode='full')
        normalized_corr = np.max(np.abs(corr_win)) / window_size
        corr_max.append(normalized_corr)
        center_time = times_array[i + window_size // 2]
        times_corr.append(center_time)
    corr_max = np.array(corr_max)
    times_corr = np.array(times_corr)

    # Create a 2x2 grid for plotting
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Phase Difference vs Time (filtered)
    axs[0, 0].plot(times_array, phase_diff, color='b')
    axs[0, 0].set_title("Phase Difference vs Time (Filtered)")
    axs[0, 0].set_xlabel("Time (s)")
    axs[0, 0].set_ylabel("Phase Difference (radians)")
    axs[0, 0].grid(True)

    # Plot 2: Sliding-window Normalized Cross-Correlation
    axs[0, 1].plot(times_corr, corr_max, color='g')
    axs[0, 1].set_title("Sliding-window Normalized Cross-Correlation")
    axs[0, 1].set_xlabel("Time (s)")
    axs[0, 1].set_ylabel("Normalized Max Correlation")
    axs[0, 1].grid(True)

    # Plot 3: Overlayed Time Series (Global Normalization)
    axs[1, 0].plot(times_array, x1_global, label="Pendulum 1", color='blue', alpha=0.7)
    axs[1, 0].plot(times_array, x2_global, label="Pendulum 2", color='orange', alpha=0.7)
    axs[1, 0].set_title("Overlayed Time Series (Global Normalization)")
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].set_ylabel("Normalized Displacement")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Plot 4: Color-coded Lissajous Plot
    sc = axs[1, 1].scatter(x1_vals, x2_vals, c=times_array, cmap='viridis', s=10)
    axs[1, 1].set_title("Color-coded Lissajous Plot")
    axs[1, 1].set_xlabel("Pendulum 1 Displacement")
    axs[1, 1].set_ylabel("Pendulum 2 Displacement")
    axs[1, 1].grid(True)
    fig.colorbar(sc, ax=axs[1, 1], label="Time (s)")

    plt.tight_layout()
    plt.savefig(r"C:\Users\avsha\Documents\cupled_pendulum_tracking\synchronization_analysis.png")
    plt.close()
    print("Saved synchronization_analysis.png")

def main():
    # Open video capture (replace with your video file or use 0 for webcam)
    cap = cv2.VideoCapture(r"C:\Users\avsha\Documents\cupled_pendulum_tracking\vid4.mp4")
    if not cap.isOpened():
        print("Error opening video stream")
        return

    scale_display = 0.75  # Use 1 for full resolution preview
    roi1, roi2, init_frame = select_rois(cap, scale_factor=scale_display)
    print("ROI selection complete.")

    # Calibration: Select two points on a fixed background element
    calib_points = select_calibration_points(init_frame)
    if len(calib_points) < 2:
        print("Not enough calibration points selected. Exiting.")
        return
    p1, p2 = calib_points[:2]
    pixel_distance = np.linalg.norm(np.array(p1) - np.array(p2))
    print("Measured pixel distance between calibration points:", pixel_distance)
    real_distance_cm = 46.3  # Example known distance in cm
    scale = real_distance_cm / pixel_distance  # cm per pixel
    print("Computed scale factor (cm per pixel):", scale)

    # Create two CSRT trackers.
    tracker1 = cv2.TrackerCSRT_create()
    tracker2 = cv2.TrackerCSRT_create()

    # Initialize the trackers with the initial frame.
    ok1 = tracker1.init(init_frame, roi1)
    ok2 = tracker2.init(init_frame, roi2)

    # Containers for tracking data.
    positions1 = []
    positions2 = []
    time_list = []
    start_time = time.time()

    cv2.namedWindow("Tracked", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tracked", 800, 600)
    print("Starting tracking. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frame captured, exiting loop.")
            break

        frame_display = cv2.resize(frame, (0, 0), fx=scale_display, fy=scale_display)
        current_time = time.time() - start_time

        ok1, bbox1 = tracker1.update(frame_display)
        ok2, bbox2 = tracker2.update(frame_display)

        if ok1:
            x1, y1, w1, h1 = bbox1
            center1 = (int(x1 + w1 / 2), int(y1 + h1 / 2))
            positions1.append(center1)
            cv2.rectangle(frame_display, (int(x1), int(y1)), (int(x1 + w1), int(y1 + h1)), (0, 255, 0), 2)
            cv2.circle(frame_display, center1, 4, (0, 255, 0), -1)
        else:
            cv2.putText(frame_display, "Tracker1 failure", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if ok2:
            x2, y2, w2, h2 = bbox2
            center2 = (int(x2 + w2 / 2), int(y2 + h2 / 2))
            positions2.append(center2)
            cv2.rectangle(frame_display, (int(x2), int(y2)), (int(x2 + w2), int(y2 + h2)), (0, 0, 255), 2)
            cv2.circle(frame_display, center2, 4, (0, 0, 255), -1)
        else:
            cv2.putText(frame_display, "Tracker2 failure", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        time_list.append(current_time)
        cv2.imshow("Tracked", frame_display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quit signal received. Ending tracking.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(time_list) < 2 or len(positions1) < 2 or len(positions2) < 2:
        print("Not enough data for analysis. Exiting.")
        return

    positions1 = np.array(positions1)
    positions2 = np.array(positions2)
    times_array = np.array(time_list)

    # For synchronization analysis, we use the x-displacements.
    x1_vals = positions1[:, 0]
    x2_vals = positions2[:, 0]
    # Optionally convert pixel displacements to real-world units:
    # x1_vals = x1_vals * scale
    # x2_vals = x2_vals * scale

    dynamic_analysis_plots(times_array, x1_vals, x2_vals, fs=30, lowcut=0.5, highcut=5.0)

if __name__ == "__main__":
    main()
