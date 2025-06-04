class Choose_Action_Physio(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        image = Image.open('Pictures//background.jpg')
        self.photo_image = ImageTk.PhotoImage(image)  # self. - for keeping the photo in memory so it will be shown
        tk.Label(self, image=self.photo_image).pack()

        exit_button_img = Image.open("Pictures//exit_system.jpg")  # Change path to your image file
        exit_button_photo = ImageTk.PhotoImage(exit_button_img)

        exit_button = tk.Button(self, image=exit_button_photo, command=lambda: self.on_click_quit(),
                                              width=exit_button_img.width, height=exit_button_img.height, bd=0, highlightthickness=0)  # Set border width to 0 to remove button border
        exit_button.image = exit_button_photo  # Store reference to image to prevent garbage collection
        exit_button.place(x=30, y=30)



         # Load images for buttons
        therapist_register_button_img = Image.open("Pictures//add_physio.jpg")  # Change path to your image file
        therapist_register_button_photo = ImageTk.PhotoImage(therapist_register_button_img)
        patient_register_button_img = Image.open("Pictures//add_patient.jpg")  # Change path to your image file
        patient_register_button_photo = ImageTk.PhotoImage(patient_register_button_img)
        go_to_training_sessions_page_button_img = Image.open("Pictures//to_patients_list.jpg")  # Change path to your image file
        go_to_training_sessions_page_button_photo = ImageTk.PhotoImage(go_to_training_sessions_page_button_img)

        go_to_training_sessions_page_button = tk.Button(self, image=go_to_training_sessions_page_button_photo,
                                                        command=lambda: self.on_go_to_training_sessions_page_click(),
                                                        width=go_to_training_sessions_page_button_img.width,
                                                        height=go_to_training_sessions_page_button_img.height,
                                                        bg='#50a6ad', bd=0,
                                                        highlightthickness=0)  # Set border width to 0 to remove button border
        go_to_training_sessions_page_button.image = go_to_training_sessions_page_button_photo  # Store reference to image to prevent garbage collection
        go_to_training_sessions_page_button.place(x=225, y=120)

        # Create buttons with images
        therapist_register_button = tk.Button(self, image=therapist_register_button_photo,
                                              command=lambda: self.on_register_physio_click(),
                                              width=therapist_register_button_img.width, height=therapist_register_button_img.height,  bg='#50a6ad', bd=0,
                                              highlightthickness=0)  # Set border width to 0 to remove button border
        therapist_register_button.image = therapist_register_button_photo  # Store reference to image to prevent garbage collection
        therapist_register_button.place(x=535, y=310)

        patient_register_button = tk.Button(self, image=patient_register_button_photo,
                                            command=lambda: self.on_register_patient_click(),
                                            width=patient_register_button_img.width, height=patient_register_button_img.height,
                                            bg='#50a6ad', bd=0,
                                            highlightthickness=0)  # Set border width to 0 to remove button border
        patient_register_button.image = patient_register_button_photo  # Store reference to image to prevent garbage collection
        patient_register_button.place(x=225, y=310)




    def on_register_physio_click(self):
        s.screen.switch_frame(PhysioRegistration)

    def on_register_patient_click(self):
        s.screen.switch_frame(PatientRegistration)


    def on_go_to_training_sessions_page_click(self):
        s.screen.switch_frame(PatientDisplaying)

    def on_click_quit(self):
        s.screen.switch_frame(ID_therapist_fill_page)

class PhysioRegistration(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        image = Image.open('Pictures//physio_registration.jpg')
        self.photo_image = ImageTk.PhotoImage(image)
        tk.Label(self, image=self.photo_image).pack()

        back_button_img = Image.open("Pictures//back_to_menu.jpg")  # Change path to your image file
        back_button_photo = ImageTk.PhotoImage(back_button_img)

        back_button = tk.Button(self, image=back_button_photo, command=lambda: self.on_click_to_physio_menu(),
                                              width=back_button_img.width, height=back_button_img.height, bd=0, highlightthickness=0)  # Set border width to 0 to remove button border
        back_button.image = back_button_photo  # Store reference to image to prevent garbage collection
        back_button.place(x=30, y=30)



        self.first_name_entry = tk.Entry(self, font=('Thaoma', 14), width=20, justify='right')
        self.first_name_entry.place(x=370, y=190)
        self.last_name_entry = tk.Entry(self, font=('Thaoma', 14), width=20, justify='right')
        self.last_name_entry.place(x=370, y=250)
        self.id_entry = tk.Entry(self, font=('Thaoma', 14), width=20, justify='right')
        self.id_entry.place(x=370, y=310)

        add_physio_button_img = Image.open("Pictures//add.jpg")  # Change path to your image file
        add_physio_button_photo = ImageTk.PhotoImage(add_physio_button_img)

        add_physio_button = tk.Button(self, image=add_physio_button_photo, command=lambda: self.on_click_physio_registration(),
                                width=add_physio_button_img.width, height=add_physio_button_img.height, bd=0,
                                highlightthickness=0)  # Set border width to 0 to remove button border
        add_physio_button.image = add_physio_button_photo  # Store reference to image to prevent garbage collection
        add_physio_button.place(x=425, y=365)
        self.labels=[] #collect the labels that apear so that on a click on the button i can delete them


    def on_click_physio_registration(self):
        self.delete_all_labels()

        excel_file_path = "Physiotherapists.xlsx"
        df = pd.read_excel(excel_file_path, sheet_name="details")
        ID_entered=self.id_entry.get()
        is_in_ID = ID_entered in df['ID'].astype(str).values #chaeck if the ID that the user inserted is already in system



        if ID_entered=="":
            back = Image.open('Pictures//no_id.jpg')
            no_id_label = ImageTk.PhotoImage(back)
            self.label = tk.Label(self, image=no_id_label, compound=tk.CENTER, highlightthickness=0)
            self.label.place(x=185, y=425)
            self.label.image = no_id_label
            self.labels.append(self.label)

        elif is_in_ID is True:
            back = Image.open('Pictures//id_already_in_system.jpg')
            id_already_in_system = ImageTk.PhotoImage(back)
            self.label = tk.Label(self, image=id_already_in_system, compound=tk.CENTER, highlightthickness=0)
            self.label.place(x=80, y=425)
            self.label.image = id_already_in_system
            self.labels.append(self.label)


        else: #insret a new row to the physio excel
            df = pd.read_excel(excel_file_path)
            new_row_data = {'ID':ID_entered , 'first name': self.first_name_entry.get(), "last name": self.last_name_entry.get()}
            new_row_df = pd.DataFrame([new_row_data])
            df = pd.concat([df, new_row_df], ignore_index=True)
            df.to_excel(excel_file_path, index=False)

            back = Image.open('Pictures//successfully_added_physio.jpg')
            successfully_added_physio = ImageTk.PhotoImage(back)
            self.label = tk.Label(self, image=successfully_added_physio, compound=tk.CENTER, highlightthickness=0)
            self.label.place(x=280, y=425)
            self.label.image = successfully_added_physio
            self.first_name_entry.delete(0, tk.END)
            self.last_name_entry.delete(0, tk.END)
            self.id_entry.delete(0, tk.END)
            self.labels.append(self.label)


    def delete_all_labels(self):
        for label in self.labels:
            label.destroy()

        self.labels = []
    def on_click_to_physio_menu(self):  # go back to the physio menu page
        s.screen.switch_frame(Choose_Action_Physio)


class PatientRegistration(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        image = Image.open('Pictures//patient_registration.jpg')
        self.photo_image = ImageTk.PhotoImage(image)
        tk.Label(self, image=self.photo_image).pack()

        back_button_img = Image.open("Pictures//back_to_menu.jpg")  # Change path to your image file
        back_button_photo = ImageTk.PhotoImage(back_button_img)

        back_button = tk.Button(self, image=back_button_photo, command=lambda: self.on_click_to_physio_menu(),
                                width=back_button_img.width, height=back_button_img.height, bd=0,
                                highlightthickness=0)  # Set border width to 0 to remove button border
        back_button.image = back_button_photo  # Store reference to image to prevent garbage collection
        back_button.place(x=30, y=30)

        self.first_name_entry = tk.Entry(self, font=('Thaoma', 14), width=20, justify='right')
        self.first_name_entry.place(x=370, y=190)
        self.last_name_entry = tk.Entry(self, font=('Thaoma', 14), width=20, justify='right')
        self.last_name_entry.place(x=370, y=250)
        self.id_entry = tk.Entry(self, font=('Thaoma', 14), width=20, justify='right')
        self.id_entry.place(x=370, y=310)

        self.options = ["גבר", "אישה"]
        self.gender = 'Male'
        self.selected_option = tk.StringVar()
        self.selected_option.set(self.options[0])  # Set the default selected option

        self.calibration_button_img = Image.open("Pictures//calibration_button.jpg")  # Change path to your image file
        self.calibration_button_photo = ImageTk.PhotoImage(self.calibration_button_img)
        self.calibration_button = tk.Button(self, image=self.calibration_button_photo, command=lambda: self.on_click_calibration(),
                                width=self.calibration_button_img.width, height=self.calibration_button_img.height, bd=0,
                                highlightthickness=0)  # Set border width to 0 to remove button border
        self.calibration_button.image = self.calibration_button_photo  # Store reference to image to prevent garbage collection
        self.calibration_button.place(x=75, y=250)



        # Create a custom style for the OptionMenu
        style = ttk.Style()
        style.theme_use('clam')  # Choose a theme (e.g., 'clam', 'default', 'alt', 'classic')

        # Configure the appearance of the OptionMenu
        style.configure('Custom.TMenubutton', font=('Arial', 12, 'bold'), background='lightgray', foreground='black',
                        anchor='e')
        style.configure('Custom.TMenubutton.TMenu', font=('Arial', 12, 'bold'), anchor='e')  # Bold font for the dropdown list

        # Create the OptionMenu with the custom style
        self.option_menu = ttk.OptionMenu(self, self.selected_option, self.selected_option.get(), *self.options,
                                          command=self.on_select_gender)
        self.option_menu['style'] = 'Custom.TMenubutton'  # Apply the custom style
        self.option_menu.config(width=6)  # Adjust the width of the grey place
        self.option_menu.place(x=440, y=365)

        self.email_entry = tk.Entry(self, font=('Thaoma', 14), width=20, justify='right')
        self.email_entry.place(x=370, y=425)

        add_patient_button_img = Image.open("Pictures//add.jpg")  # Change path to your image file
        add_patient_button_photo = ImageTk.PhotoImage(add_patient_button_img)

        add_patient_button = tk.Button(self, image=add_patient_button_photo,
                                      command=lambda: self.on_click_patient_registration(),
                                      width=add_patient_button_img.width, height=add_patient_button_img.height, bd=0,
                                      highlightthickness=0)  # Set border width to 0 to remove button border
        add_patient_button.image = add_patient_button_photo  # Store reference to image to prevent garbage collection
        add_patient_button.place(x=425, y=480)
        self.labels = []  # collect the labels that apear so that on a click on the button i can delete them

    def on_select_gender(self, option):
        if option=='אישה':
            self.gender='Female'

        else:
            self.gender='Male'

    def on_click_calibration(self):
        s.asked_for_measurement = True

        # Remove existing calibration label if it exists
        if hasattr(self, 'cal_lable') and self.cal_lable.winfo_exists():
            self.cal_lable.destroy()

        self.cal_lable = tk.Label(
            self,
            text="עמוד מול המצלמה\nוהצמד ידיים לצידי הגוף",
            compound=tk.CENTER,
            highlightthickness=0,
            justify=tk.CENTER,  # Align text to the right
            fg="red",  # Set text color to red
            font=("Arial", 16, "bold")  # Set font to Arial, size 16, bold
        )
        self.cal_lable.place(x=30, y=370)
        self.labels.append(self.cal_lable)

        # Load alternate image
        self.calibration_button_img_pressed = Image.open("Pictures//doing_calibration.jpg")
        self.calibration_button_photo_pressed = ImageTk.PhotoImage(self.calibration_button_img_pressed)

        self.calibration_button.config(image=self.calibration_button_photo_pressed)
        self.calibration_button.image = self.calibration_button_photo_pressed
        self.after(8000, self.perform_calibration)

    def perform_calibration(self):

        while s.average_dist is None:
            time.sleep(0.1)

        # Determine the message based on whether calibration was successful
        if s.average_dist == -1:  # Means it didn't recognize enough keypoints
            new_text = 'הכיול לא צלח\nהקפד לעמוד מול המצלמה\nולחץ שנית על "כייל"'
        else:
            new_text = f"{str(round(s.average_dist, 2))} :מרחק בין הכתפיים"

        # Check if the label already exists
        if hasattr(self, 'cal_lable') and self.cal_lable.winfo_exists():
            # Update existing label text instead of creating a new one
            self.cal_lable.config(text=new_text)
        else:
            # Create the label only if it doesn't exist
            self.cal_lable = tk.Label(
                self,
                text=new_text,
                compound=tk.CENTER,
                highlightthickness=0,
                justify=tk.CENTER,  # Align text to the right
                fg="red",  # Set text color to red
                font=("Arial", 16, "bold")  # Set font to Arial, size 16, bold
            )
            self.cal_lable.place(x=30, y=450)
            self.labels.append(self.cal_lable)

        # Restore the button image after calibration
        self.calibration_button.config(image=self.calibration_button_photo)
        self.calibration_button.image = self.calibration_button_photo

    def is_email_valid (self, email):
        try:
            # Validate the email address
            validate_email(email)
            # If no exception is raised, the email is valid
            return True
        except EmailNotValidError:
            # If an exception is raised, the email is not valid
            #print(str(e))
            return False

    def on_click_patient_registration(self):
        self.delete_all_labels()
        excel_file_path = "Patients.xlsx"
        workbook=openpyxl.load_workbook(excel_file_path)
        df = pd.read_excel(excel_file_path, sheet_name="patients_details")
        ID_entered=self.id_entry.get()
        is_in_ID = ID_entered in df['ID'].astype(str).values #chaeck if the ID that the user inserted is already in system



        if ID_entered=="":
            back = Image.open('Pictures//no_id.jpg')
            background_img = ImageTk.PhotoImage(back)

            self.label = tk.Label(self, image=background_img, compound=tk.CENTER, highlightthickness=0)
            self.label.place(x=185, y=490)
            self.label.image = background_img
            self.labels.append(self.label)


        elif is_in_ID is True:
            error = Image.open('Pictures//id_already_in_system.jpg')
            id_already_in_system = ImageTk.PhotoImage(error)
            self.label = tk.Label(self, image=id_already_in_system, compound=tk.CENTER, highlightthickness=0)
            self.label.place(x=80, y=490)
            self.label.image = id_already_in_system
            self.labels.append(self.label)


        elif self.is_email_valid(self.email_entry.get()) is False: #if email is not valid
            error = Image.open('Pictures//not_valid_email.jpg')
            id_already_in_system = ImageTk.PhotoImage(error)
            self.label = tk.Label(self, image=id_already_in_system, compound=tk.CENTER, highlightthickness=0)
            self.label.place(x=230, y=490)
            self.label.image = id_already_in_system
            self.labels.append(self.label)

        elif s.average_dist is None:
            error = Image.open('Pictures//didnt_do_calibration.jpg')
            id_already_in_system = ImageTk.PhotoImage(error)
            self.label = tk.Label(self, image=id_already_in_system, compound=tk.CENTER, highlightthickness=0)
            self.label.place(x=20, y=490)
            self.label.image = id_already_in_system
            self.labels.append(self.label)


        else:
            s.chosen_patient_ID=ID_entered
            #modifying the columns that has other value than false
            new_row_data_details={
                'ID': str(ID_entered),
                'first name': str(self.first_name_entry.get()),
                'last name': str(self.last_name_entry.get()),
                'gender': self.gender,
                'number of exercises': 25,
                'email': str(self.email_entry.get()),
                'number of repetitions in each exercise' : 10,
                'rate': "moderate",
                "email of therapist": "yaelszn@gmail.com",
                'dist between shoulders': s.average_dist
            }

            s.average_dist = None
            sheet1=workbook["patients_details"]
            columns = list(new_row_data_details.keys())
            sheet1.append([new_row_data_details[column] for column in columns])
            #workbook.save("Patients.xlsx")

            #insert a row to the excel of history of training
            new_row_hystory_of_training = {'ID': ID_entered}
            sheet2 = workbook["patients_history_of_trainings"]
            columns = list(new_row_hystory_of_training.keys())
            sheet2.append([new_row_hystory_of_training[column] for column in columns])
            #workbook.save("Patients.xlsx")

            #insert a row to the excel of exercises
            df2 = pd.read_excel(excel_file_path, sheet_name="patients_exercises")
            sheet3 = workbook["patients_exercises"]
            new_row_data_exercises = {column: True for column in df2.columns}
            new_row_data_exercises.update({'ID': ID_entered})

            columns = list(new_row_data_exercises.keys())
            sheet3.append([new_row_data_exercises[column] for column in columns])
            workbook.save("Patients.xlsx")

            self.create_folders_when_insert_patient()

            back = Image.open('Pictures//successfully_added_patient.jpg')
            successfully_added_patient = ImageTk.PhotoImage(back)
            self.label = tk.Label(self, image=successfully_added_patient, compound=tk.CENTER, highlightthickness=0)
            self.label.place(x=170, y=490)
            self.label.image = successfully_added_patient
            self.first_name_entry.delete(0, tk.END)
            self.last_name_entry.delete(0, tk.END)
            self.id_entry.delete(0, tk.END)
            self.selected_option.set(self.options[0])
            self.email_entry.delete(0, tk.END)
            self.gender='Male'
            self.labels.append(self.label)


    def create_folders_when_insert_patient(self):
        Excel.create_and_open_folder(f"Patients/{s.chosen_patient_ID}")  # open folder for patient
        Excel.create_and_open_folder(f"Patients/{s.chosen_patient_ID}/Graphs")  # open graphs folder
        Excel.create_and_open_folder(f"Patients/{s.chosen_patient_ID}/Tables")  # open graphs folder

        df = pd.read_excel("Patients.xlsx", sheet_name="patients_exercises", header=None)

        # Assuming the headers are in the first row of the DataFrame
        headers = df.iloc[0]  # Extract headers from the first row

        # Exclude a specific header
        header_to_exclude = 'ID'
        selected_headers = [header for header in headers if header != header_to_exclude]

        # Iterate over each selected header
        for header in selected_headers:
            #create a file for each execise
            Excel.create_and_open_folder(f"Patients/{s.chosen_patient_ID}/Graphs/{header}")  # open graphs folder
            Excel.create_and_open_folder(f"Patients/{s.chosen_patient_ID}/Tables/{header}")

    def delete_all_labels(self):
        for label in self.labels:
            label.destroy()

        self.labels = []
    def on_click_to_physio_menu(self):  # go back to the physio menu page
        s.screen.switch_frame(Choose_Action_Physio)
