import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class ImageRotator:
    def __init__(self):
        pass
    
    def load_image(self, image_path):
        """Load image from file path"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        return image
    
    def detect_skew_angle(self, image):
        """Detect the skew angle using edge detection and Hough Line Transform"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection using Canny
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        
        # Detect lines using Hough Transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
        
        if lines is None:
            print("No lines detected for skew calculation")
            return 0
        
        # Calculate angles from detected lines
        angles = []
        for line in lines[:20]:  # Consider first 20 lines
            rho, theta = line[0]
            angle = np.degrees(theta)
            
            # Convert angle to be relative to horizontal
            if angle > 90:
                angle = angle - 180
            elif angle < -90:
                angle = angle + 180
                
            # Filter out vertical lines (around 90 degrees)
            if abs(angle) < 45:
                angles.append(angle)
        
        if not angles:
            print("No suitable lines found for angle calculation")
            return 0
        
        # Return median angle to avoid outliers
        median_angle = np.median(angles)
        print(f"Detected angles: {angles[:5]}...")  # Show first 5 angles
        print(f"Median skew angle: {median_angle:.2f} degrees")
        
        return median_angle
    
    def rotate_image(self, image, angle):
        """Rotate image by given angle (positive = counterclockwise, negative = clockwise)"""
        if abs(angle) < 0.1:
            print("Angle too small, skipping rotation")
            return image
        
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new image dimensions to fit entire rotated image
        cos_val = abs(rotation_matrix[0, 0])
        sin_val = abs(rotation_matrix[0, 1])
        new_width = int((height * sin_val) + (width * cos_val))
        new_height = int((height * cos_val) + (width * sin_val))
        
        # Adjust rotation matrix for translation to center the image
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        # Perform rotation with white background
        rotated = cv2.warpAffine(
            image, rotation_matrix, (new_width, new_height), 
            flags=cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_CONSTANT, 
            borderValue=(255, 255, 255)
        )
        
        return rotated
    
    def auto_straighten(self, image):
        """Automatically detect and correct skew"""
        angle = self.detect_skew_angle(image)
        corrected_angle = -angle  # Negative to correct the skew
        print(f"Applying correction angle: {corrected_angle:.2f} degrees")
        return self.rotate_image(image, corrected_angle)
    
    def manual_rotate(self, image, angle):
        """Manually rotate image by specified angle"""
        print(f"Applying manual rotation: {angle:.2f} degrees")
        return self.rotate_image(image, angle)
    
    def show_comparison(self, original, rotated, title="Image Rotation"):
        """Display original and rotated images side by side"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original image
        axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image', fontsize=14)
        axes[0].axis('off')
        
        # Rotated image
        axes[1].imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Rotated Image', fontsize=14)
        axes[1].axis('off')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def save_image(self, image, output_path):
        """Save the rotated image"""
        success = cv2.imwrite(output_path, image)
        if success:
            print(f"Image saved to: {output_path}")
        else:
            print(f"Failed to save image to: {output_path}")
        return success
    
    def process_image(self, image_path, output_path=None, manual_angle=None, show_result=True):
        """Main function to process image rotation"""
        try:
            # Load image
            print(f"Loading image: {image_path}")
            original = self.load_image(image_path)
            print(f"Image shape: {original.shape}")
            
            # Rotate image
            if manual_angle is not None:
                rotated = self.manual_rotate(original, manual_angle)
            else:
                rotated = self.auto_straighten(original)
            
            # Save if output path provided
            if output_path:
                self.save_image(rotated, output_path)
            
            # Show comparison
            if show_result:
                rotation_type = f"Manual ({manual_angle}°)" if manual_angle is not None else "Auto-corrected"
                self.show_comparison(original, rotated, f"Image Rotation - {rotation_type}")
            
            return {
                'original': original,
                'rotated': rotated,
                'success': True
            }
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return {'success': False, 'error': str(e)}

# Convenience functions for direct use
def rotate_image_auto(image_path, output_path=None):
    """Auto-detect and correct skew in image"""
    rotator = ImageRotator()
    return rotator.process_image(image_path, output_path, manual_angle=None)

def rotate_image_manual(image_path, angle, output_path=None):
    """Manually rotate image by specified angle"""
    rotator = ImageRotator()
    return rotator.process_image(image_path, output_path, manual_angle=angle)

def batch_rotate_images(folder_path, output_folder=None, angle=None):
    """Rotate multiple images in a folder"""
    rotator = ImageRotator()
    
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return
    
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Supported formats
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    processed_files = []
    for filename in os.listdir(folder_path):
        if any(filename.lower().endswith(ext) for ext in extensions):
            image_path = os.path.join(folder_path, filename)
            
            if output_folder:
                name, ext = os.path.splitext(filename)
                output_path = os.path.join(output_folder, f"{name}_rotated{ext}")
            else:
                output_path = None
            
            print(f"\nProcessing: {filename}")
            result = rotator.process_image(
                image_path, output_path, manual_angle=angle, show_result=False
            )
            
            if result['success']:
                processed_files.append(filename)
    
    print(f"\nSuccessfully processed {len(processed_files)} images")
    return processed_files

def create_skewed_dataset(straightened_folder, output_folder, rotation_angles=None, angle_range=(-15, 15), save_labels=True):
    """
    Create skewed images from straightened images with known rotation angles
    
    Args:
        straightened_folder: Path to folder with straight images
        output_folder: Path to save rotated images
        rotation_angles: List of specific angles to use, or None for random
        angle_range: Tuple of (min_angle, max_angle) for random rotation
        save_labels: Whether to save ground truth angles to a text file
    """
    import random
    import json
    
    rotator = ImageRotator()
    
    # Check input folder
    if not os.path.exists(straightened_folder):
        print(f"Input folder not found: {straightened_folder}")
        return []
    
    # Create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")
    
    # Supported formats
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    # Get all image files
    image_files = [f for f in os.listdir(straightened_folder) 
                   if any(f.lower().endswith(ext) for ext in extensions)]
    
    if not image_files:
        print(f"No image files found in {straightened_folder}")
        return []
    
    # Prepare rotation angles
    if rotation_angles is None:
        # Generate random angles
        rotation_angles = [random.uniform(angle_range[0], angle_range[1]) 
                          for _ in range(len(image_files))]
    elif len(rotation_angles) < len(image_files):
        # Repeat angles if not enough provided
        rotation_angles = (rotation_angles * ((len(image_files) // len(rotation_angles)) + 1))[:len(image_files)]
    
    processed_results = []
    ground_truth_labels = {}
    
    print(f"Creating skewed dataset from {len(image_files)} straightened images")
    print("=" * 60)
    
    for i, (filename, angle) in enumerate(zip(image_files, rotation_angles), 1):
        input_path = os.path.join(straightened_folder, filename)
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}_skewed_{angle:.1f}{ext}"
        output_path = os.path.join(output_folder, output_filename)
        
        print(f"[{i}/{len(image_files)}] {filename} -> Skewing by {angle:.2f}°")
        
        try:
            # Load straight image
            straight_image = rotator.load_image(input_path)
            
            # Apply rotation to create skewed image
            skewed_image = rotator.rotate_image(straight_image, angle)
            
            # Save skewed image
            success = rotator.save_image(skewed_image, output_path)
            
            if success:
                processed_results.append({
                    'original_file': filename,
                    'skewed_file': output_filename,
                    'rotation_angle': angle,
                    'status': 'success'
                })
                
                # Store ground truth (negative angle because we need correction angle)
                ground_truth_labels[output_filename] = {
                    'applied_rotation': angle,
                    'correction_angle': -angle,  # Angle needed to straighten
                    'original_file': filename
                }
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            processed_results.append({
                'original_file': filename,
                'skewed_file': '',
                'rotation_angle': angle,
                'status': 'failed',
                'error': str(e)
            })
    
    # Save ground truth labels
    if save_labels and ground_truth_labels:
        labels_file = os.path.join(output_folder, 'ground_truth_labels.json')
        with open(labels_file, 'w') as f:
            json.dump(ground_truth_labels, f, indent=2)
        print(f"\nGround truth labels saved to: {labels_file}")
        
        # Also save as CSV for easy reading
        csv_file = os.path.join(output_folder, 'ground_truth_labels.csv')
        with open(csv_file, 'w') as f:
            f.write("skewed_filename,original_filename,applied_rotation,correction_angle\n")
            for skewed_file, data in ground_truth_labels.items():
                f.write(f"{skewed_file},{data['original_file']},{data['applied_rotation']},{data['correction_angle']}\n")
        print(f"Ground truth CSV saved to: {csv_file}")
    
    # Print summary
    successful = len([r for r in processed_results if r['status'] == 'success'])
    print(f"\n{'='*60}")
    print(f"DATASET CREATION SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully processed: {successful}/{len(image_files)} images")
    print(f"Output folder: {output_folder}")
    print(f"Ground truth saved: {'Yes' if save_labels else 'No'}")
    
    return processed_results

def batch_rotate_with_predefined_angles(input_folder, output_folder, angle_list):
    """
    Rotate images with specific predefined angles
    
    Args:
        input_folder: Folder with input images
        output_folder: Folder to save rotated images  
        angle_list: List of angles to apply to images in order
    """
    rotator = ImageRotator()
    
    if not os.path.exists(input_folder):
        print(f"Input folder not found: {input_folder}")
        return []
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_files = [f for f in os.listdir(input_folder) 
                   if any(f.lower().endswith(ext) for ext in extensions)]
    
    if not image_files:
        print(f"No images found in {input_folder}")
        return []
    
    results = []
    print(f"Rotating {len(image_files)} images with predefined angles")
    
    for i, filename in enumerate(image_files):
        # Use angle from list (cycle if list is shorter)
        angle = angle_list[i % len(angle_list)]
        
        input_path = os.path.join(input_folder, filename)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_folder, f"{name}_rot_{angle}{ext}")
        
        print(f"{filename} -> {angle}° rotation")
        
        result = rotator.process_image(input_path, output_path, 
                                     manual_angle=angle, show_result=False)
        
        if result['success']:
            results.append({'file': filename, 'angle': angle, 'status': 'success'})
        else:
            results.append({'file': filename, 'angle': angle, 'status': 'failed'})
    
    return results

# Example usage
if __name__ == "__main__":
    # Example 1: Auto-detect and correct skew
    # rotate_image_auto('license_plate.jpg', 'straightened.jpg')
    
    # Example 2: Manual rotation
    # rotate_image_manual('license_plate.jpg', -5.2, 'rotated.jpg')
    
    # Example 3: Batch processing
    # batch_rotate_images('input_folder/', 'output_folder/', angle=-3)
    
    # Interactive example
    rotator = ImageRotator()
    
    # Replace with your image path
    image_path = "your_image.jpg"
    
    if os.path.exists(image_path):
        # Try auto-detection first
        print("=== AUTO ROTATION ===")
        result_auto = rotator.process_image(image_path)
        
        # Try manual rotation
        print("\n=== MANUAL ROTATION ===")
        result_manual = rotator.process_image(image_path, manual_angle=-3)
    else:
        print("Please update the image_path variable with your actual image file")
        print("\nExample usage:")
        print("rotator = ImageRotator()")
        print("rotator.process_image('your_image.jpg', 'output.jpg', manual_angle=-5)")
