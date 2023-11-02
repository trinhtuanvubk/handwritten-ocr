import cv2
import numpy as np
import os
import time

def increase_contrast_with_clahe(img):

    # Convert to YUV color space
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # Create a CLAHE object
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(6, 6))

    # Apply CLAHE to the Luminance channel
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])

    # Convert back to BGR color space
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    return img_output


def remove_lines(image):
    # Load the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to isolate the lines and text
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    # Detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    # Combine the detected horizontal and vertical lines
    detect_lines = cv2.addWeighted(detect_horizontal, 0.5, detect_vertical, 0.5, 0)

    # Find contours of the lines
    cnts = cv2.findContours(detect_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Draw over the detected lines with white color
    for c in cnts:
        cv2.drawContours(image, [c], -1, (255,255,255), 2)

    # Inpaint the original image to reconstruct the area where the lines were removed
    result = cv2.inpaint(image, detect_lines, 3, cv2.INPAINT_TELEA)

    return result

def detect_text_lines(img):

    img_ori = img.copy()
    img = increase_contrast_with_clahe(img)

    # Remove lines from the image
    img = remove_lines(img)

    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Removing Shadows
    dilated_img = cv2.dilate(img, np.ones((7,7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(img, bg_img)
    img = diff_img

    # Apply blur to smooth out the edges
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Apply threshold to get image with only b&w (binarization)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Morphological operations to assist in isolating lines of text
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.dilate(img, kernel, iterations=4)
    img = cv2.erode(img, kernel, iterations=18)

    # Find contours of potential text lines
    contours, hierarchy = cv2.findContours(255 - img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    W_img = img.shape[1]
    H_img = img.shape[0]

    box_contour_list = []
    # Filter and draw rectangles on potential text lines
    for contour in contours:
        # Get rectangle bounding contour
        [x, y, w, h] = cv2.boundingRect(contour)

        # Don't plot small false positives that aren't text
        if h < 0.2 * H_img or w > 0.2 * W_img or w < 0.004 * W_img:
            continue

        box_contour_list.append([x, y, x + w, y + h])
        # Draw rectangle around contour on original image
        # cv2.rectangle(img_ori, (x, y), (x + w, y + h), (255, 0, 255), 2)

    box_contour_list = sorted(box_contour_list, key=lambda box: box[1])   # sort top to botton
    merge_box_lines = []
    current_line = [box_contour_list[0]]

    for box in box_contour_list[1:]:
        _, current_line_bottom, _, _ = current_line[-1]
        top = box[1]

        # Merge boxes that are vertically aligned (i.e., their y-coordinates overlap)
        if abs(top - current_line_bottom)  <=  0.15 * H_img:  # You might want to adjust the threshold here
            current_line.append(box)
        else:
            merge_box_lines.append(current_line)
            current_line = [box]

    merge_box_lines.append(current_line)

    boxes_final = []

    if len(merge_box_lines) > 0:
    # for line in merge_box_lines:
        # if len(line) > 3 or len(merge_box_lines) == 1:
            # get line longest
            line =  max(merge_box_lines, key=len)
            xs = [box[0] for box in line]
            ys = [box[1] for box in line]
            x2s = [box[2] for box in line]
            y2s = [box[3] for box in line]

            min_x = min(xs)
            min_y = min(ys)
            max_x = max(x2s)
            max_y = max(y2s)

            # Calculate the width and height of the box
            box_width = max_x - min_x
            box_height = max_y - min_y

            # Extend the box by 20%
            extend_width = box_width * 0.1
            extend_height = box_height * 0.5

            # Update the coordinates with the extensions while ensuring they are within the image bounds
            min_x = max(0, int(min_x - extend_width // 2))
            min_y = max(0, int(min_y - extend_height // 2))
            max_x = min(W_img, int(max_x + extend_width // 2))
            max_y = min(H_img, int(max_y + extend_height // 2))

            # boxes_final.append([int(min_x), int(min_y), int(max_x), int(max_y)])   # format xmin ymin xmax ymax
            # cv2.rectangle(img_ori, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (255, 0, 0), 2)
            result_img = img_ori[ min_y:max_y, min_x: max_x]
    else:
        result_img = img_ori
    # Return the img line croped or img origin
    return result_img



if __name__ == "__main__":

    folder_out = "Dataset/debug"
    folder_in  = "/home/sonlt373/Desktop/SoNg/OCR_handwriting_shop_sticker/dev/handwritten-ocr/Dataset/checker"
    for path in os.listdir(folder_in):
        img_path = os.path.join(folder_in, path)
        img = cv2.imread(img_path)

        t0 = time.time()

        processed_img = detect_text_lines(img)

        # Show the image with detected text lines
        # cv2.imshow('Detected Text Lines', processed_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        print('[time perimg]=====>', time.time() - t0, "(s)")
        # Save the image with detected text lines
        cv2.imwrite(os.path.join(folder_out, path), processed_img)
        # break
