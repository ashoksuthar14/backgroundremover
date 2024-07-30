import streamlit as st
import cv2
import numpy as np
from rembg import remove
from PIL import Image
import io

st.title("Background Removal App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an opencv image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

    # Check if the image has an alpha channel
    if image.shape[2] == 4:
        b, g, r, a = cv2.split(image)
        rgb_image = cv2.merge((b, g, r))
    else:
        rgb_image = image

    # Convert the image to grayscale
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

    # Create a binary mask where white areas are set to 255 and the rest to 0
    _, binary_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    # Invert the binary mask
    binary_mask_inv = cv2.bitwise_not(binary_mask)

    # Create a mask with 3 channels
    mask_rgb = cv2.merge([binary_mask_inv, binary_mask_inv, binary_mask_inv])

    # Apply the mask to the RGB image
    result = cv2.bitwise_and(rgb_image, mask_rgb)

    # Add alpha channel to the result
    b, g, r = cv2.split(result)
    a = binary_mask_inv
    final_result = cv2.merge((b, g, r, a))

    # Use rembg to further remove background
    img_pil = Image.fromarray(cv2.cvtColor(final_result, cv2.COLOR_BGRA2RGBA))
    result_img_pil = remove(img_pil)

    # Convert to displayable format
    result_img_bytes = io.BytesIO()
    result_img_pil.save(result_img_bytes, format='PNG')
    result_img_bytes = result_img_bytes.getvalue()

    # Display the original image
    st.image(image, caption='Original Image', use_column_width=True)

    # Display the result
    st.image(result_img_bytes, caption='Image with Background Removed', use_column_width=True)

    # Provide download link
    st.download_button(
        label="Download image",
        data=result_img_bytes,
        file_name="output.png",
        mime="image/png"
    )
