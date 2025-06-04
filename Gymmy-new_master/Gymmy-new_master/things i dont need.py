def find_and_change_values_effort_in_Last(headers_row=1):
    # Select the desired sheet
    sheet = s.Last_workbook["effort"]
    new_values_dict = s.list_effort_each_exercise

    # Get all the keys (exercise names) from the new_values_dict
    #search_values = list(new_values_dict.keys())

    # Check if the sheet is empty (contains only headers)
    if sheet.max_row == headers_row:
        # Add headers
        sheet.cell(row=headers_row, column=1, value="exercise")
        sheet.cell(row=headers_row, column=2, value="effort")

    # Iterate over new_values_dict
    for exercise_name, effort_number in new_values_dict.items():
        found = False
        # Iterate through the existing rows to check if the exercise is already in the sheet
        for row in sheet.iter_rows(min_row=headers_row + 1, max_row=sheet.max_row, min_col=1, max_col=2):
            if row[0].value == exercise_name:
                found = True
                # Update the value in the second column
                row[1].value = effort_number
                break

        # If the exercise is not found in the sheet, add it as a new row
        if not found:
            new_row_index = sheet.max_row + 1
            sheet.cell(row=new_row_index, column=1, value=exercise_name)
            sheet.cell(row=new_row_index, column=2, value=effort_number)


def find_and_change_values_success_in_Last(headers_row=1):
    # Select the desired sheet
    sheet = s.Last_workbook["success"]
    new_values_dict = s.ex_list

    # Get all the keys (exercise names) from the new_values_dict
    #search_values = list(new_values_dict.keys())

    # Check if the sheet is empty (contains only headers)
    if sheet.max_row == headers_row:
        # Add headers
        sheet.cell(row=headers_row, column=1, value="exercise")
        sheet.cell(row=headers_row, column=2, value="number of successful repetitions")

    # Iterate over new_values_dict
    for exercise_name, success_count in new_values_dict.items():
        found = False
        # Iterate through the existing rows to check if the exercise is already in the sheet
        for row in sheet.iter_rows(min_row=headers_row + 1, max_row=sheet.max_row, min_col=1, max_col=2):
            if row[0].value == exercise_name:
                found = True
                # Update the value in the second column
                row[1].value = success_count
                break

        # If the exercise is not found in the sheet, add it as a new row
        if not found:
            new_row_index = sheet.max_row + 1
            sheet.cell(row=new_row_index, column=1, value=exercise_name)
            sheet.cell(row=new_row_index, column=2, value=success_count)


def extract_string_between_spaces(input_string):
    parts = input_string.split()  # Split the string into parts based on spaces
    if len(parts) >= 3:  # Ensure there are at least three parts (two spaces)
        return parts[1]  # Extract the second part
    else:
        return None


def wait_until_waving(self):
    s.waved_has_tool = False
    s.req_exercise = "hello_waving"
    while not s.waved_has_tool:
        time.sleep(0.00000001)




def add_daytime(to_say):
    hour = datetime.now().hour
    print(hour)

    if (22 <= hour <= 23) or (0 <= hour <= 5):
        to_say += "לילה טוב!"

    if (6 <= hour <= 10):
        to_say += "בוקר טוב!"

    if (11 <= hour <= 14):
        to_say += "צהריים טובים!"

    if (15 <= hour <= 18):
        to_say += "אחר צהריים טובים!"

    if (19 <= hour <= 21):
        to_say += "ערב טוב!"

    return to_say


def text_to_speech(text, lang='iw', slow=False):
    tts = gTTS(text=text, lang=lang, slow=slow)
    tts.save("output.mp3")

    # Initialize pygame mixer
    pygame.mixer.init()

    # Load and play the audio file
    pygame.mixer.music.load("output.mp3")
    pygame.mixer.music.play()

    # Wait for the audio to finish playing
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    # Clean up
    pygame.mixer.quit()



def text_to_speech2(language='iw'):
    text = "הרם את הידיים ב-90 מעלות"
    tts = gTTS(text=text, lang=language, slow=False, lang_check=False)
    tts.save("output.mp3")

    # Play the generated speech using a media player
    os.system("start output.mp3")





def two_angles_graph(df, exercise):
    worksheet = s.excel_workbook.add_worksheet(("graphs_"+exercise)[:31])

    first_graph_name = df.iloc[0, 0] + ", " + df.iloc[4, 0] + ", " + df.iloc[8, 0]
    y_values_1 = df.iloc[48, :]
    y_values_1_float = y_values_1.astype(float)
    create_and_save_graph(df.columns, y_values_1_float, first_graph_name, worksheet, 48)

    second_graph_name = df.iloc[12, 0] + ", " + df.iloc[16, 0] + ", " + df.iloc[20, 0]
    y_values_2 = df.iloc[49, :]
    y_values_2_float = y_values_2.astype(float)
    create_and_save_graph(df.columns, y_values_2_float, second_graph_name, worksheet, 49)

    third_graph_name = df.iloc[24, 0] + ", " + df.iloc[28, 0] + ", " + df.iloc[32, 0]
    y_values_3 = df.iloc[50, :]
    y_values_3_float = y_values_3.astype(float)
    create_and_save_graph(df.columns, y_values_3_float, third_graph_name, worksheet, 50)

    fourth_graph_name = df.iloc[36, 0] + ", " + df.iloc[40, 0] + ", " + df.iloc[44, 0]
    y_values_4 = df.iloc[51, :]
    y_values_4_float = y_values_4.astype(float)
    create_and_save_graph(df.columns, y_values_4_float, fourth_graph_name, worksheet, 51)





def three_angles_graph(df, exercise):
    worksheet = s.excel_workbook.add_worksheet(("graphs_"+exercise)[:31])

    first_graph_name = df.iloc[0, 0] + ", " + df.iloc[4, 0] + ", " + df.iloc[8, 0]
    y_values_1 = df.iloc[72, :]
    y_values_1_float = y_values_1.astype(float)
    create_and_save_graph(df.columns, y_values_1_float, first_graph_name, worksheet, 72)

    second_graph_name = df.iloc[12, 0] + ", " + df.iloc[16, 0] + ", " + df.iloc[20, 0]
    y_values_2 = df.iloc[73, :]
    y_values_2_float = y_values_2.astype(float)
    create_and_save_graph(df.columns, y_values_2_float, second_graph_name, worksheet, 73)

    third_graph_name = df.iloc[24, 0] + ", " + df.iloc[28, 0] + ", " + df.iloc[32, 0]
    y_values_3 = df.iloc[74, :]
    y_values_3_float = y_values_3.astype(float)
    create_and_save_graph(df.columns, y_values_3_float, third_graph_name, worksheet, 74)

    fourth_graph_name = df.iloc[36, 0] + ", " + df.iloc[40, 0] + ", " + df.iloc[44, 0]
    y_values_4 = df.iloc[75, :]
    y_values_4_float = y_values_4.astype(float)
    create_and_save_graph(df.columns, y_values_4_float, fourth_graph_name, worksheet, 75)

    fifth_graph_name = df.iloc[48, 0] + ", " + df.iloc[52, 0] + ", " + df.iloc[56, 0]
    y_values_5 = df.iloc[76, :]
    y_values_5_float = y_values_5.astype(float)
    create_and_save_graph(df.columns, y_values_5_float, fifth_graph_name, worksheet, 76)

    sixth_graph_name = df.iloc[60, 0] + ", " + df.iloc[64, 0] + ", " + df.iloc[68, 0]
    y_values_6 = df.iloc[77, :]
    y_values_6_float = y_values_6.astype(float)
    create_and_save_graph(df.columns, y_values_6_float, sixth_graph_name, worksheet, 77)



# EX14 - Hands behind the head and turn to each side
    def hands_behind_and_turn_both_sides_notool(self, i): #לא עשיתי
        if i==0:
            self.gymmy.r_shoulder_x.goto_position(-40, 1.5, wait=False)
            self.gymmy.l_shoulder_x.goto_position(40, 1.5, wait=False)
            self.gymmy.r_shoulder_y.goto_position(-180, 1.5, wait=False)
            self.gymmy.l_shoulder_y.goto_position(-180, 1.5, wait=False)
            self.gymmy.r_arm_z.goto_position(80, 1.5, wait=False)
            self.gymmy.l_arm_z.goto_position(-80, 1.5, wait=False)
            self.gymmy.r_arm[3].goto_position(-60, 1.5, wait=False)
            self.gymmy.l_arm[3].goto_position(-60, 1.5, wait=False)

        self.gymmy.abs_z.goto_position(-180, 2, wait=False)
        time.sleep(3)
        self.gymmy.abs_z.goto_position(-100, 2, wait=False)
        time.sleep(3)
        self.gymmy.abs_z.goto_position(-20, 2, wait=False)
        time.sleep(3)
        self.gymmy.abs_z.goto_position(-100, 2, wait=False)
        time.sleep(3)


        if i == (s.rep - 1):
            # init
            self.gymmy.abs_z.goto_position(-100, 1.5, wait=False)
            self.gymmy.r_arm[3].goto_position(90, 1.5, wait=False)
            self.gymmy.l_arm[3].goto_position(90, 1.5, wait=False)
            self.gymmy.l_arm_z.goto_position(0, 1.5, wait=False)
            self.gymmy.r_arm_z.goto_position(0, 1.5, wait=False)
            self.gymmy.l_shoulder_y.goto_position(0, 1.5, wait=False)
            self.gymmy.r_shoulder_y.goto_position(0, 1.5, wait=False)
            self.gymmy.l_shoulder_x.goto_position(0, 1.5, wait=False)
            self.gymmy.r_shoulder_x.goto_position(0, 1.5, wait=False)


#search for previous training that the exercise was in (before the last one that was found)
def search_for_previous_graphs_of_exercise(exercise_name, last_training_exercise_was_in):
        # Load the workbook
        file_path = "Patients.xlsx"
        workbook = openpyxl.load_workbook(file_path)

        # Select the desired sheet
        sheet = workbook["patients_history_of_trainings"]
        row_of_patient=""

        # Iterate through the rows to find the value in the first column
        for row in sheet.iter_rows(min_row=2, max_row=sheet.max_row, min_col=1, max_col=1):
            cell = row[0]
            if str(cell.value) == s.chosen_patient_ID:
                row_of_patient=row
                break  # Stop searching after finding the value


        #if this is the first page of shown exercises
        if last_training_exercise_was_in == "":
            # Exclude the first cell and select only the even-indexed cells
            for cell in reversed(row_of_patient[1::2]):
                exercise_existing = check_worksheet_exists(cell, ("graphs_" + exercise_name)[:31])
                if exercise_existing:
                    return cell

        else:
            #if there was one page or more with graphs shown before
            found_previous = False
            for cell in reversed(row_of_patient[1::2]):
                if cell.value == last_training_exercise_was_in:
                    found_previous=True

                if found_previous:
                    exercise_existing= check_worksheet_exists(cell, ("graphs_"+exercise_name)[:31])
                    if exercise_existing:
                        return cell




def check_worksheet_exists(workbook_path, worksheet_name):
    try:

        # Load the workbook
        workbook = openpyxl.load_workbook(f"{s.chosen_patient_ID}/{workbook_path}.xlsx")

        # Check if the worksheet exists
        if worksheet_name in workbook.sheetnames:
            print(f"Worksheet '{worksheet_name}' exists in the workbook.")
            return True
        else:
            print(f"Worksheet '{worksheet_name}' does not exist in the workbook.")
            return False

    except Exception as e:
        print(f"An error occurred: {e}")
        return False