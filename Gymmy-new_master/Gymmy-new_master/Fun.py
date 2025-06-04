import os
import re
import math
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
import Excel
import Settings as s


def get_ordered_images(image_list):
    """
    Reorder the image list according to the pattern requested:
    - 2 images: 1, 2
    - 4 images: 1, 3, 2, 4
    - 6 images: 1, 4, 2, 5, 3, 6
    """
    n = len(image_list)
    if n == 2:
        return [image_list[0], image_list[1]]
    elif n == 4:
        return [image_list[0], image_list[2], image_list[1], image_list[3]]
    elif n == 6:
        return [image_list[0], image_list[3], image_list[1], image_list[4], image_list[2], image_list[5]]
    else:
        return image_list  # If the number of images is not 2, 4, or 6, return as-is


def collect_images_from_folders(patient_id, exercise_name, timestamp):
    """
    Collect all the image file paths from the 'Graphs' and 'Tables' folders for a given exercise.
    The images are ordered by their number suffix as described.
    """
    # Define the base paths for the graphs and tables folders
    base_path_graphs = f"Patients/{patient_id}/Graphs/{exercise_name}/{timestamp}"
    base_path_tables = f"Patients/{patient_id}/Tables/{exercise_name}/{timestamp}"

    # Collect all .jpeg images from the Graphs folder and .png images from the Tables folder
    graph_images = sorted([os.path.join(base_path_graphs, img) for img in os.listdir(base_path_graphs) if img.endswith('.jpeg')], key=lambda x: int(re.search(r'(\d+)\.jpeg$', x).group(1)))
    table_images = sorted([os.path.join(base_path_tables, img) for img in os.listdir(base_path_tables) if img.endswith('.png')], key=lambda x: int(re.search(r'(\d+)\.png$', x).group(1)))

    # Combine and reorder the images
    graph_images_ordered = get_ordered_images(graph_images)
    table_images_ordered = get_ordered_images(table_images)

    return table_images_ordered, graph_images_ordered  # Tables first, then Graphs


def create_pdf(pdf_filename, image_groups, section_headers, global_header_line1, global_header_line2, global_header_line3):
    """
    Create a PDF with a global header, section headers, and fixed-size images, with the images
    ordered for specific layouts depending on the number of images in each section.
    """
    # Register the Hebrew-compatible font
    pdfmetrics.registerFont(TTFont('Hebrew', "C:/Users/yaels/יעל פרוייקט גמר/zedcheck/arial.ttf-master/arial.ttf"))

    # Create a canvas to generate the PDF
    pdf_canvas = canvas.Canvas(pdf_filename, pagesize=letter)

    # Set the PDF page dimensions (letter size)
    width, height = letter

    # Calculate the total height needed for the global header (3 lines of text with intervals)
    line_height = 0.5 * inch  # Height of each line plus some interval
    total_header_height = 3 * line_height

    # Calculate the starting Y position to vertically center the global header
    start_y_position = (height + total_header_height) / 2 + 100  # This centers the 3 lines of text

    # Set the font for the global header
    pdf_canvas.setFont("Hebrew", 24)

    # Line 1
    text_width_line1 = pdf_canvas.stringWidth(global_header_line1, "Hebrew", 24)
    x_position_line1 = (width - text_width_line1) / 2  # Center the first line horizontally
    pdf_canvas.drawString(x_position_line1, start_y_position, global_header_line1)

    # Line 2 (with interval)
    text_width_line2 = pdf_canvas.stringWidth(global_header_line2, "Hebrew", 24)
    x_position_line2 = (width - text_width_line2) / 2  # Center the second line horizontally
    pdf_canvas.drawString(x_position_line2, start_y_position - 1.5 * line_height, global_header_line2)

    # Line 3 (with interval)
    text_width_line3 = pdf_canvas.stringWidth(global_header_line3, "Hebrew", 24)
    x_position_line3 = (width - text_width_line3) / 2  # Center the third line horizontally
    pdf_canvas.drawString(x_position_line3, start_y_position - 3 * line_height, global_header_line3)

    # Start placing content after the header
    current_y_position = start_y_position - 3 * line_height - inch  # Adjust Y position after the header

    # Set a consistent padding/margin between images and sections
    image_padding = 0.1 * inch  # Reduced padding between images
    section_padding = 0.25 * inch  # Padding between sections

    # Calculate the image size based on a 3-images-per-row layout
    images_per_row = 3
    margin_x = 0.4 * inch  # Reduced margin to make images bigger
    image_width = (width - 2 * margin_x - (images_per_row - 1) * image_padding) / images_per_row
    image_height = image_width * 2 / 3  # Maintain aspect ratio (3:2) for the images

    # Iterate over each section, placing the section header and corresponding images
    for section_index, image_paths in enumerate(image_groups):
        # Draw the section header only once
        section_header_printed = False

        # Determine the number of images in the current section
        total_images = len(image_paths)

        if total_images == 12:
            # Split the image list into two halves
            first_half = image_paths[:6]
            second_half = image_paths[6:]

            # Reorder the images to match the requested layout for 12 images
            reordered_images = [
                first_half[0], first_half[1], first_half[2],  # First row
                second_half[0], second_half[1], second_half[2],  # Second row
                first_half[3], first_half[4], first_half[5],  # Third row
                second_half[3], second_half[4], second_half[5]  # Fourth row
            ]
            images_per_row = 3
        elif total_images == 8:
            # Split the image list into two halves
            first_half = image_paths[:4]
            second_half = image_paths[4:]

            # Reorder the images to match the requested layout for 8 images
            reordered_images = [
                first_half[0], first_half[1],  # First row
                second_half[0], second_half[1],  # Second row
                first_half[2], first_half[3],  # Third row
                second_half[2], second_half[3]  # Fourth row
            ]
            images_per_row = 2  # For layout, but image size remains based on 3 per row
        elif total_images == 4:
            # Reorder the images to match the requested layout for 4 images
            reordered_images = [
                image_paths[0], image_paths[1],  # First row
                image_paths[2], image_paths[3]  # Second row
            ]
            images_per_row = 2  # For layout, but image size remains based on 3 per row
        else:
            # If the image count doesn't match the specific cases, keep the original order
            reordered_images = image_paths
            images_per_row = 3 if total_images >= 3 else total_images

        # Calculate the height needed for this section (including images and section header)
        rows_in_section = math.ceil(len(reordered_images) / images_per_row)
        total_section_height = rows_in_section * (image_height + image_padding) + section_padding + 20  # 20 for header

        # Check if there's enough space to fit the entire section on the current page
        if current_y_position - total_section_height < inch:
            # Not enough space, move to the next page
            pdf_canvas.showPage()  # Create a new page
            current_y_position = height - 1 * inch  # Reset Y position for the new page (starting from the top)
            section_header_printed = False  # Ensure the header is printed once when the new page is started

        # Print section header and images
        for i, image_path in enumerate(reordered_images):
            # Print the section header once before the first image
            if not section_header_printed:
                # Draw the section header
                section_header = section_headers[section_index]
                pdf_canvas.setFont("Hebrew", 14)  # Use Hebrew font for section headers
                pdf_canvas.drawString(1 * inch, current_y_position, section_header)
                section_header_printed = True  # Mark that the section header has been printed
                current_y_position -= section_padding  # Adjust Y position after section header

            # Calculate the total width of images in the current row to center them
            row_start_index = i // images_per_row * images_per_row  # Starting index of the current row
            images_in_row = min(images_per_row, len(reordered_images) - row_start_index)
            total_row_width = images_in_row * image_width + (images_in_row - 1) * image_padding
            x_start_position = (width - total_row_width) / 2  # Center the row of images

            # Calculate x and y positions for the image
            x_position = x_start_position + (i % images_per_row) * (image_width + image_padding)  # Adjust x to center images
            y_position_offset = (i // images_per_row) * (image_height + image_padding)  # Adjust y based on row index

            # Insert the image at the calculated position
            pdf_canvas.drawImage(image_path, x_position, current_y_position - y_position_offset - image_height,
                                 width=image_width, height=image_height)

            # Draw a border around the image
            pdf_canvas.rect(x_position, current_y_position - y_position_offset - image_height,
                            image_width, image_height, stroke=1, fill=0)

        # Adjust Y position for the next section
        current_y_position -= rows_in_section * (image_height + image_padding) + 50

    # Finalize the PDF
    pdf_canvas.save()

# Example usage
exercises = ['bend_elbows_ball', 'open_arms_and_forward_ball']
image_groups = []
section_headers = []

s.chosen_patient_ID = "314808981"
s.starts_and_ends_of_stops = ['17-10-2024 14-23-33']
patient_workbook_path = "Patients.xlsx"
first_name = Excel.find_value_by_colName_and_userID(patient_workbook_path, "patients_details", s.chosen_patient_ID,
                                                    "first name")
last_name = Excel.find_value_by_colName_and_userID(patient_workbook_path, "patients_details", s.chosen_patient_ID,
                                                   "last name")


start_time = s.starts_and_ends_of_stops[-1]

date_part, time_part = start_time.split(' ')
formatted_date = date_part.replace('-', '/')
formatted_time = time_part.replace('-', ':')
formatted_dt = f'{formatted_date} {formatted_time}'

# Global header with 3 lines
global_header_line1 = f' סיכום אימון של המטופל: {first_name} {last_name}'[::-1]
global_header_line2 = f'מספר מטופל: {s.chosen_patient_ID[::-1]}'[::-1]
global_header_line3 = f'זמן האימון: {formatted_dt[::-1]}'[::-1]


for exercise_name in exercises:
    # Collect the images for each exercise
    table_images, graph_images = collect_images_from_folders(s.chosen_patient_ID, exercise_name, s.starts_and_ends_of_stops[-1])

    # Add images and section headers to the lists (Tables first, then Graphs)
    image_groups.append(table_images + graph_images)  # Add all images for the exercise
    section_headers.append(f"Exercise name: {exercise_name.capitalize()}")  # Section header for the exercise


# Define the directory path where you want to save the PDF
output_directory = f'Patients/{s.chosen_patient_ID}/'

# Check if the directory exists, if not, create it
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Define the full path including the filename
output_path = os.path.join(output_directory, f'{start_time}.pdf')

# Create the PDF with images, headers, and a global title
create_pdf(output_path, image_groups, section_headers, global_header_line1, global_header_line2, global_header_line3)
