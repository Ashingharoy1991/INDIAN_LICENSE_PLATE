import easyocr
import cv2
from ultralytics import YOLO
import numpy as np
import os
from datetime import datetime
from collections import defaultdict, Counter
from smart_plate_cleaner import SmartPlateTextCleaner
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='License Plate Recognition System')
    parser.add_argument('--video', required=True, help='Path to the input video file')
    parser.add_argument('--model', default='license_plate_detector.pt', help='Path to YOLO model weights')
    parser.add_argument('--frame_dir', default='detection_frames', help='Directory to save detection frames')
    parser.add_argument('--plate_dir', default='recognized_plates', help='Directory to save recognized plates')
    parser.add_argument('--min_predictions', type=int, default=5, help='Minimum predictions needed for consensus')
    parser.add_argument('--consensus_threshold', type=float, default=0.6, help='Minimum ratio for consensus (0-1)')
    parser.add_argument('--max_predictions', type=int, default=5, help='Maximum predictions to store per track')
    parser.add_argument('--brightness', type=int, default=100, help='Brightness threshold for plate detection')
    parser.add_argument('--show', action='store_true', help='Show real-time detection window')
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Initialize YOLO license plate detector
    model = YOLO(args.model)

    # Initialize EasyOCR with custom model
    reader = easyocr.Reader(['en'], recog_network='custom_example', gpu=True)

    # Create directories for saving results
    os.makedirs(args.frame_dir, exist_ok=True)
    os.makedirs(args.plate_dir, exist_ok=True)

    # Tracking configuration
    MIN_PREDICTIONS_FOR_CONSENSUS = args.min_predictions
    CONSENSUS_CONFIDENCE_THRESHOLD = args.consensus_threshold
    MAX_PREDICTIONS_PER_TRACK = args.max_predictions
    BRIGHTNESS_THRESHOLD = args.brightness

    # Storage for tracking predictions
    track_predictions = defaultdict(list)  # {track_id: [{'text': str, 'confidence': float, 'frame': int}, ...]}
    finalized_plates = {}  # {track_id: {'text': str, 'confidence': float, 'frame_first_seen': int}}
    processed_tracks = set()  # Track IDs that have been processed and saved

    def preprocess_plate(plate_img):
        img = cv2.resize(plate_img, (333, 75))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img_binary = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        img_erode = cv2.erode(img_binary, (3,3))
        img_dilate = cv2.dilate(img_erode, (3,3))
        
        return img_dilate

    def clean_plate_text(text):
        """Clean and format recognized license plate text - IMPROVED VERSION"""
        cleaner = SmartPlateTextCleaner()
        result = cleaner.clean_with_validation(text)
        
        # You can add additional logging here if needed
        if result['confidence'] < 0.8:
            print(f"Warning: Low confidence cleaning for '{text}' -> '{result['cleaned']}'")
        
        return result['cleaned']

    def get_consensus_prediction(predictions):
        """
        Get the most common prediction from a list of predictions
        
        Args:
            predictions: List of dicts with 'text', 'confidence', 'frame' keys
        
        Returns:
            dict with consensus prediction or None if no consensus
        """
        if len(predictions) < MIN_PREDICTIONS_FOR_CONSENSUS:
            return None
        
        # Count occurrences of each text
        text_counts = Counter([pred['text'] for pred in predictions])
        most_common_text, most_common_count = text_counts.most_common(1)[0]
        
        # Check if we have enough consensus
        consensus_ratio = most_common_count / len(predictions)
        if consensus_ratio < CONSENSUS_CONFIDENCE_THRESHOLD:
            return None
        
        # Get average confidence for the most common prediction
        matching_predictions = [pred for pred in predictions if pred['text'] == most_common_text]
        avg_confidence = sum(pred['confidence'] for pred in matching_predictions) / len(matching_predictions)
        first_frame = min(pred['frame'] for pred in matching_predictions)
        
        return {
            'text': most_common_text,
            'confidence': avg_confidence,
            'count': most_common_count,
            'total_predictions': len(predictions),
            'consensus_ratio': consensus_ratio,
            'frame_first_seen': first_frame
        }

    def add_prediction(track_id, text, confidence, frame_num):
        """Add a new prediction for a track ID"""
        prediction = {
            'text': text,
            'confidence': confidence,
            'frame': frame_num
        }
        
        track_predictions[track_id].append(prediction)
        
        # Limit the number of stored predictions per track
        if len(track_predictions[track_id]) > MAX_PREDICTIONS_PER_TRACK:
            track_predictions[track_id].pop(0)  # Remove oldest prediction
        
        print(f"Track {track_id}: Added prediction '{text}' (conf: {confidence:.2f}) - Total: {len(track_predictions[track_id])}")

    def should_finalize_track(track_id):
        """Check if a track should be finalized based on prediction count"""
        predictions = track_predictions[track_id]
        return len(predictions) >= MIN_PREDICTIONS_FOR_CONSENSUS

    def finalize_track_prediction(track_id):
        """Finalize the prediction for a track ID using consensus"""
        if track_id in processed_tracks:
            return None
            
        predictions = track_predictions[track_id]
        consensus = get_consensus_prediction(predictions)
        
        if consensus:
            # Apply cleaning function to the consensus prediction
            cleaned_text = clean_plate_text(consensus['text'])
            
            final_prediction = {
                'original_text': consensus['text'],
                'cleaned_text': cleaned_text,
                'confidence': consensus['confidence'],
                'count': consensus['count'],
                'total_predictions': consensus['total_predictions'],
                'consensus_ratio': consensus['consensus_ratio'],
                'frame_first_seen': consensus['frame_first_seen']
            }
            
            finalized_plates[track_id] = final_prediction
            processed_tracks.add(track_id)
            
            print(f"Track {track_id}: FINALIZED - '{consensus['text']}' -> '{cleaned_text}' "
                f"(consensus: {consensus['count']}/{consensus['total_predictions']}, "
                f"ratio: {consensus['consensus_ratio']:.2f})")
            
            return final_prediction
        
        return None

    # Open video file
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video: {args.video}")
        exit()

    frame_count = 0
    recognition_log = []

    print(f"Starting video processing...")
    print(f"Configuration:")
    print(f"- Min predictions for consensus: {MIN_PREDICTIONS_FOR_CONSENSUS}")
    print(f"- Consensus confidence threshold: {CONSENSUS_CONFIDENCE_THRESHOLD}")
    print(f"- Max predictions per track: {MAX_PREDICTIONS_PER_TRACK}")
    print(f"- Brightness threshold: {BRIGHTNESS_THRESHOLD}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (2560, 1440))
        frame_count += 1
        
        # Tracking with BOTSORT
        results = model.track(
            frame,
            conf=0.6,
            tracker="botsort.yaml",
            persist=True,
            verbose=False
        )
        
        for r in results:
            if r.boxes.id is None:
                continue  # Skip frames with no detections
            
            boxes = r.boxes
            track_ids = boxes.id.cpu().numpy().astype(int)
            xyxys = boxes.xyxy.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            
            for tid, (x1, y1, x2, y2), conf in zip(track_ids, xyxys, confs):
                # Skip if this track is already processed
                if tid in processed_tracks:
                    # Draw the finalized result
                    if tid in finalized_plates:
                        final_text = finalized_plates[tid]['cleaned_text']
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f'ID:{tid} {final_text} [FINAL]', 
                                (x1, max(y1 - 10, 10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    continue
                
                # Ensure coordinates are within frame bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                
                plate_crop = frame[y1:y2, x1:x2]
                if plate_crop.size == 0:
                    continue
                    
                # Resize crop plate
                # plate_crop = cv2.resize(plate_crop, (333, 75))

                # Check brightness
                avg_brightness = np.mean(plate_crop)
                if avg_brightness < BRIGHTNESS_THRESHOLD:
                    continue
                
                # Perform OCR
                ocr_results = reader.recognize(plate_crop)
                
                if ocr_results:
                    for bbox, text, prob in ocr_results:
                        if prob > 0.8 and len(text) >= 9:  # Confidence threshold
                            # Add prediction to tracking
                            add_prediction(tid, text, prob, frame_count)
                            
                            # Draw current prediction (not finalized)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)  # Orange for collecting
                            prediction_count = len(track_predictions[tid])
                            cv2.putText(frame, f'ID:{tid} {text} [{prediction_count}/{MIN_PREDICTIONS_FOR_CONSENSUS}]', 
                                    (x1, max(y1 - 10, 10)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                            
                            # Check if we should finalize this track
                            if should_finalize_track(tid):
                                final_prediction = finalize_track_prediction(tid)
                                
                                if final_prediction:
                                    # Save results for finalized prediction
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                                    clean_text = final_prediction['cleaned_text']
                                    
                                    # Save the original frame with detection
                                    frame_filename = f"{args.frame_dir}/frame_{timestamp}_ID{tid}_{clean_text}.jpg"
                                    cv2.imwrite(frame_filename, frame)

                                    # Save the cropped plate image
                                    plate_filename = f"{args.plate_dir}/ID{tid}_{clean_text}.jpg"
                                    cv2.imwrite(plate_filename, plate_crop)

                                    # Log the recognition
                                    recognition_log.append({
                                        'track_id': tid,
                                        'frame': frame_count,
                                        'timestamp': timestamp,
                                        'original_text': final_prediction['original_text'],
                                        'cleaned_text': clean_text,
                                        'confidence': final_prediction['confidence'],
                                        'consensus_count': final_prediction['count'],
                                        'total_predictions': final_prediction['total_predictions'],
                                        'consensus_ratio': final_prediction['consensus_ratio'],
                                        'frame_path': frame_filename,
                                        'plate_path': plate_filename
                                    })
                                    
                                    print(f"SAVED: Track {tid} - {final_prediction['original_text']} -> {clean_text}")

        # Display the frame
        cv2.imshow("License Plate Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Process any remaining tracks that haven't been finalized
    print("\nProcessing remaining tracks...")
    for track_id in list(track_predictions.keys()):
        if track_id not in processed_tracks:
            final_prediction = finalize_track_prediction(track_id)
            if final_prediction:
                # Log the final prediction even if not saved during video processing
                recognition_log.append({
                    'track_id': track_id,
                    'frame': final_prediction['frame_first_seen'],
                    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
                    'original_text': final_prediction['original_text'],
                    'cleaned_text': final_prediction['cleaned_text'],
                    'confidence': final_prediction['confidence'],
                    'consensus_count': final_prediction['count'],
                    'total_predictions': final_prediction['total_predictions'],
                    'consensus_ratio': final_prediction['consensus_ratio'],
                    'frame_path': 'N/A',
                    'plate_path': 'N/A'
                })

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    # Save recognition log
    log_filename = f"recognition_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(log_filename, 'w') as f:
        f.write("TrackID,Frame,Timestamp,OriginalText,CleanedText,Confidence,ConsensusCount,TotalPredictions,ConsensusRatio,FramePath,PlatePath\n")
        for entry in recognition_log:
            f.write(f"{entry['track_id']},{entry['frame']},{entry['timestamp']},{entry['original_text']},"
                    f"{entry['cleaned_text']},{entry['confidence']:.2f},{entry['consensus_count']},"
                    f"{entry['total_predictions']},{entry['consensus_ratio']:.2f},{entry['frame_path']},{entry['plate_path']}\n")

    print(f"\nProcessing complete!")
    print(f"Recognition log saved to {log_filename}")
    print(f"Total tracks processed: {len(processed_tracks)}")
    print(f"Total finalized plates: {len(finalized_plates)}")

    # Print summary of finalized plates
    print("\nFinalized Plates Summary:")
    for track_id, plate_info in finalized_plates.items():
        print(f"Track {track_id}: {plate_info['original_text']} -> {plate_info['cleaned_text']} "
            f"(consensus: {plate_info['count']}/{plate_info['total_predictions']})")

if __name__ == "__main__":
    main()