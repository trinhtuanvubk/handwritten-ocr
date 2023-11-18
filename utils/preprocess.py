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

    if len(box_contour_list)==0:
        return None
    
    # Sort via height =============
    # box_contour_list = sorted(box_contour_list, key=lambda box: box[1])   # sort top to botton
    # merge_box_lines = []
    # current_line = [box_contour_list[0]]

    # for box in box_contour_list[1:]:
    #     current_line_left, current_line_bottom, w_, h_ = current_line[-1]
    #     top = box[1]
    #     left = box[0]

    #     # Merge boxes that are vertically aligned (i.e., their y-coordinates overlap)
    #      # You might want to adjust the threshold here
    #     if (abs(top - current_line_bottom)  <=  0.15 * H_img) :
    #                 # and (abs(left - (current_line_left + w_)) <= 0.15 * W_img):
    #         current_line.append(box)
    #     else:
    #         merge_box_lines.append(current_line)
    #         current_line = [box]

    # merge_box_lines.append(current_line)
    # ================================

    # Sort via width ==============
    box_contour_list = sorted(box_contour_list, key=lambda box: box[0])   # sort top to botton
    # max_width_ele = max([(box[3]-box[0]) for box in box_contour_list])
    merge_box_lines = []
    current_line = [box_contour_list[0]]

    for box in box_contour_list[1:]:
        current_line_left, current_line_top, current_line_right, current_line_bottom = current_line[-1]
        left, top , right, bottom = box

        # You might want to adjust the threshold here
        if (abs(left - current_line_right) <= 0.15 * W_img):
            # (bottom >= current_line_top):
            current_line.append(box)
        else:
            # In this kalapa challenge: just have only one sentence
            continue
        # else:
        #     merge_box_lines.append(current_line)
        #     current_line = [box]

    merge_box_lines.append(current_line)
    # ================================

    # Sort and only check the last
    # box_contour_list = sorted(box_contour_list, key=lambda box: box[0]) 
    # merge_box_lines = []
    # last_line_left, _, last_line_right, _ = box_contour_list[-1]
    # second_last_line_left, _, second_last_line_right, _ = box_contour_list[-2]
    # if (abs(last_line_left - second_last_line_right) <= 0.15 * W_img):
    #     merge_box_lines.append(box_contour_list)
    # else:
    #     merge_box_lines.append(box_contour_list[:-1])

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
            extend_width_left = box_width * 0.1
            extend_width_right = box_width * 0.1
            extend_height = box_height * 1

            # Update the coordinates with the extensions while ensuring they are within the image bounds
            min_x = max(0, int(min_x - extend_width_left))
            # min_x = 0
            min_y = max(0, int(min_y - extend_height // 2))
            # min_y = 0
            max_x = min(W_img, int(max_x + extend_width_right // 2))
            max_y = min(H_img, int(max_y + extend_height // 2))

            # boxes_final.append([int(min_x), int(min_y), int(max_x), int(max_y)])   # format xmin ymin xmax ymax
            # cv2.rectangle(img_ori, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (255, 0, 0), 2)
            result_img = img_ori[ min_y:max_y, min_x: max_x]
    else:
        result_img = img_ori
    # Return the img line croped or img origin
    # return result_img
    return result_img

def detect_text_lines_v2(img, num_words):

    def top_k_nosort(arr, topk=10):
        top_k_max_values = sorted(arr, reverse=True, key=lambda i: ((i[2]-i[0])*2+(i[3]-i[1])))[:topk]
        result_list = [value for value in arr if value in top_k_max_values]
        result_index = [i for i,value in enumerate(arr) if value in top_k_max_values]
        return result_list, result_index

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

    if len(box_contour_list)==0:
        return None
    
    # Sort via width ==============
    box_contour_list = sorted(box_contour_list, key=lambda box: box[0])   # sort top to botton

    if len(box_contour_list)==num_words:
        for box in box_contour_list:
            cv2.rectangle(img_ori, (int(box[0]), 5), (int(box[2]), int(H_img-5)), (255, 0, 0), 2)
        result_img = img_ori
        return result_img
    
    if len(box_contour_list) < num_words:
        return None


    topk_contour, topk_index = top_k_nosort(box_contour_list, topk= num_words)

    # max_width_ele = max([(box[3]-box[0]) for box in box_contour_list])
    # top_num_word_box = sorted(box_contour_list, key=lambda box:((box[2]-box[0])+(box[3]-box[1])))[:num_words]
    merge_box_lines = []
    # current_line = [box_contour_list[0]]
    curr_topk_idx = 0
    for i, box in enumerate(box_contour_list):
        # current_line_left, current_line_top, current_line_right, current_line_bottom = current_line[-1]
        if i in topk_index:
            if curr_topk_idx<len(topk_index)-1:
                curr_topk_idx += 1
            else:
                continue
        else:
            left, top , right, bottom = box
            print(curr_topk_idx)
            curr_left, curr_top , curr_right, curr_bottom = topk_contour[curr_topk_idx]
            last_curr_left, last_curr_top , last_curr_right, last_curr_bottom = topk_contour[curr_topk_idx-1]

            if curr_topk_idx == 0:
                topk_contour[curr_topk_idx][0] = min(left, curr_left)
                topk_contour[curr_topk_idx][2] = max(right, curr_right)

            elif curr_topk_idx == len(topk_contour):
                topk_contour[curr_topk_idx][0] = min(left, curr_left)
                topk_contour[curr_topk_idx][2] = max(right, curr_right)
            else:
                # front_dis =  abs((curr_left + (curr_right-curr_left)/2) - (left + (right-left)/2))
                # back_dis =  abs((last_curr_left + (last_curr_right-last_curr_left)/2) - (left + (right-left)/2))
                front_dis = curr_left - right
                back_dis = left - last_curr_right
                if front_dis<=back_dis:
                    topk_contour[curr_topk_idx][0] = min(left, curr_left)
                    topk_contour[curr_topk_idx][2] = max(right, curr_right)
                else:
                    topk_contour[curr_topk_idx-1][0] = min(left, last_curr_left)
                    topk_contour[curr_topk_idx-1][2] = max(right, last_curr_right)

    for i, box in enumerate(topk_contour):
        if i==0:
            box[0] = max(0, box[0] - box[0]/5*2)
            box[2] = box[2] + (topk_contour[i+1][0] - box[2])/5*2
        elif i==len(topk_contour)-1:
            box[0] = box[0] - (box[0] - topk_contour[i-1][2])/5*2
            box[2] = min(W_img, box[2] + (W_img - box[2])/5*2)
        else:
            box[0] = box[0] - (box[0] - topk_contour[i-1][2])/7*3
            box[2] = box[2] + (topk_contour[i+1][0] - box[2])/5*2
        cv2.rectangle(img_ori, (int(box[0]), 3), (int(box[2]), int(H_img-3)), (255, 0, 0), 2)

    result_img = img_ori
    return result_img


# if __name__ == "__main__":

#     folder_out = "Dataset/debug"
#     folder_in  = "/home/sonlt373/Desktop/SoNg/OCR_handwriting_shop_sticker/dev/handwritten-ocr/Dataset/checker"
#     for path in os.listdir(folder_in):
#         img_path = os.path.join(folder_in, path)
#         img = cv2.imread(img_path)

#         t0 = time.time()

#         processed_img = detect_text_lines(img)

#         # Show the image with detected text lines
#         # cv2.imshow('Detected Text Lines', processed_img)
#         # cv2.waitKey(0)
#         # cv2.destroyAllWindows()
#         print('[time perimg]=====>', time.time() - t0, "(s)")
#         # Save the image with detected text lines
#         cv2.imwrite(os.path.join(folder_out, path), processed_img)
#         # break
