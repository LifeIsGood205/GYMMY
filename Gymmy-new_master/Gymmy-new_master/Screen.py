# -*- coding: utf-8 -*-
import time
import tkinter as tk
from PIL import Image, ImageTk
import Settings as s
from Audio import say
import random

class Screen(tk.Tk):
    def __init__(self):
        print("screen start")
        tk.Tk.__init__(self, className='Gymmy')
        self._frame = None
        self["bg"]="#F3FCFB"
        self.user_id_entry = tk.Entry(self, font=('Arial', 14), width=20)
        self.user_id_entry.pack(pady=10)

    def switch_frame(self, frame_class):
        """Destroys current frame and replaces it with a new one."""
        new_frame = frame_class(self)
        if self._frame is not None:
            if hasattr(self._frame, 'background_label'):
                self._frame.background_label.destroy()
            self._frame.destroy()
        self._frame = new_frame
        self._frame.pack()

    # def switchFromtryAgaintoBlank(self):
    #     self._frame.background_label.destroy()
    #     s.screen.switch_frame(BlankPage)


    def wait_until_waving(self):
        s.req_exercise = "hello_waving"
        while not s.waved_has_tool:
            time.sleep(0.00000001)

    def quit(self):
        self.destroy()

class HelloPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        image = Image.open('Pictures//hello.jpg')
        self.photo_image = ImageTk.PhotoImage(image) #self. - for keeping the photo in memory so it will be shown
        tk.Label(self, image = self.photo_image).pack()
        say('hello_waving')




class ChooseExercise(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        image1 = Image.open('Pictures//choose_exercise.jpg')
        self.photo_image1 = ImageTk.PhotoImage(image1)
        self.background_label = tk.Label(image=self.photo_image1)
        self.background_label.pack()
        say('choose_exercise')

        image2 = Image.open('Pictures//choose_ball1.jpg')
        self.photo_image2 = ImageTk.PhotoImage(image2)
        button2= tk.Button(image=self.photo_image2, command=lambda:self.on_click_ball1(master))
        button2.pack()
        button2.place(x=520, y=340)

        image3 = Image.open('Pictures//choose_ball2.jpg')
        self.photo_image3 = ImageTk.PhotoImage(image3)
        button2= tk.Button(image=self.photo_image3, command=lambda:self.on_click_ball2(master))
        button2.pack()
        button2.place(x=315, y=340)

        image4 = Image.open('Pictures//choose_band.jpg')
        self.photo_image4 = ImageTk.PhotoImage(image4)
        button2 = tk.Button(image=self.photo_image4, command=lambda:self.on_click_band(master))
        button2.pack()
        button2.place(x=640, y=150)

        image5 = Image.open('Pictures//choose_stick.jpg')
        self.photo_image5 = ImageTk.PhotoImage(image5)
        button2 = tk.Button(image=self.photo_image5, command=lambda:self.on_click_stick(master))
        button2.pack()
        button2.place(x=430, y=150)

        image6 = Image.open('Pictures//choose_noTool.jpg')
        self.photo_image6 = ImageTk.PhotoImage(image6)
        button2 = tk.Button(image=self.photo_image6, command=lambda:self.on_click_noTool(master))
        button2.pack()
        button2.place(x=225, y=150)
        say("choose_exercise")


    def on_click_ball1(self, master):
        print("image clicked")
        s.clicked = True
        s.ball1 = True
        s.screen.switch_frame(Ball1)
       # self.after(0, lambda: master.switch_frame(Ball1))


    def on_click_ball2(self, master):
        print("image clicked")
        s.clicked = True
        s.ball2 = True
        s.screen.switch_frame(Ball2)
        #self.after(0, lambda: master.switch_frame(Ball2))

    def on_click_band(self, master):
        print("image clicked")
        s.clicked = True
        s.band = True
        s.screen.switch_frame(Band)
        #self.after(0, lambda: master.switch_frame(RubberBand))


    def on_click_stick(self, master):
        print("image clicked")
        s.clicked = True
        s.stick = True
        s.screen.switch_frame(Stick)
        #self.after(0, lambda: master.switch_frame(Stick))


    def on_click_noTool(self, master):
        print("image clicked")
        s.clicked = True
        s.noTool = True
        s.waved_has_tool = False
        say("noTool")
        s.ex_in_training = ["open_arms_with_rubber_band", "open_arms_and_up_with_rubber_band",
                            "up_with_rubber_band_and_turn"]
        self.after(9000, lambda: master.wait_until_waving())

        #self.after(0, lambda: master.switch_frame(ExercisePage))

class Ball1(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        image = Image.open('Pictures//ball.jpg')
        self.photo_image = ImageTk.PhotoImage(image) #self. - for keeping the photo in memory so it will be shown
        tk.Label(self, image = self.photo_image).pack()
        s.waved_has_tool = False
        say("Ball1")
        s.ex_in_training=["bend_elbows_ball", "raise_arms_above_head_ball", "raise_arms_forward_turn_ball"]
        self.after(9000,lambda: master.wait_until_waving())




class Ball2(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        image = Image.open('Pictures//ball.jpg')
        self.photo_image = ImageTk.PhotoImage(image) #self. - for keeping the photo in memory so it will be shown
        tk.Label(self, image = self.photo_image).pack()
        s.waved_has_tool = False
        say("Ball2")
        s.ex_in_training = ["open_arms_and_forward_ball", "open_arms_above_head_ball"]
        self.after(9000, lambda: master.wait_until_waving())


class Stick(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        image = Image.open('Pictures//stick.jpg')
        self.photo_image = ImageTk.PhotoImage(image) #self. - for keeping the photo in memory so it will be shown
        tk.Label(self, image = self.photo_image).pack()
        s.waved_has_tool = False
        say("Stick")
        s.ex_in_training = ["bend_elbows_stick", "bend_elbows_and_up_stick", "arms_up_and_down_stick", "switch_with_stick"]
        self.after(9000, lambda: master.wait_until_waving())


class RubberBand(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        image = Image.open('Pictures//rubber_band.jpg')
        self.photo_image = ImageTk.PhotoImage(image) #self. - for keeping the photo in memory so it will be shown
        tk.Label(self, image = self.photo_image).pack()
        s.waved_has_tool = False
        say("Band")
        s.ex_in_training = ["open_arms_with_rubber_band", "open_arms_and_up_with_rubber_band", "up_with_rubber_band_and_turn"]
        self.after(9000, lambda: master.wait_until_waving())


class EffortScale(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        image_scale = Image.open('Pictures/scale.jpg')
        self.photo_image = ImageTk.PhotoImage(image_scale)
        self.background_label = tk.Label(image=self.photo_image)
        self.background_label.pack()

        image0 = Image.open('Pictures//scale_0.jpg')
        self.photo_image0 = ImageTk.PhotoImage(image0)
        button2= tk.Button(image=self.photo_image0, command=self.on_click_0)
        button2.pack()
        button2.place(height=47, width=400, x=540, y=150)

        image1 = Image.open('Pictures//scale_1.jpg')
        self.photo_image1 = ImageTk.PhotoImage(image1)
        button2= tk.Button(image=self.photo_image1, command=self.on_click_1)
        button2.pack()
        button2.place(height=47, width=400, x=540, y=197)

        image2 = Image.open('Pictures//scale_2.jpg')
        self.photo_image2 = ImageTk.PhotoImage(image2)
        button2 = tk.Button(image=self.photo_image2, command=self.on_click_2)
        button2.pack()
        button2.place(height=47, width=400, x=540, y=244)

        image3 = Image.open('Pictures//scale_3.jpg')
        self.photo_image3 = ImageTk.PhotoImage(image3)
        button2 = tk.Button(image=self.photo_image3, command=self.on_click_3)
        button2.pack()
        button2.place(height=47, width=400, x=540, y=291)

        image4 = Image.open('Pictures//scale_4.jpg')
        self.photo_image4 = ImageTk.PhotoImage(image4)
        button2 = tk.Button(image=self.photo_image4, command=self.on_click_4)
        button2.pack()
        button2.place(height=47, width=400, x=540, y=338)

        image5 = Image.open('Pictures//scale_5.jpg')
        self.photo_image5 = ImageTk.PhotoImage(image5)
        button2 = tk.Button(image=self.photo_image5, command=self.on_click_5)
        button2.pack()
        button2.place(height=47, width=400, x=540, y=385)

        image6 = Image.open('Pictures//scale_6.jpg')
        self.photo_image6 = ImageTk.PhotoImage(image6)
        button2 = tk.Button(image=self.photo_image6, command=self.on_click_6)
        button2.pack()
        button2.place(height=47, width=400, x=540, y=431)

        image7 = Image.open('Pictures//scale_7.jpg')
        self.photo_image7 = ImageTk.PhotoImage(image7)
        button2 = tk.Button(image=self.photo_image7, command=self.on_click_7)
        button2.pack()
        button2.place(height=47, width=400, x=100, y=150)

        image8 = Image.open('Pictures//scale_8.jpg')
        self.photo_image8 = ImageTk.PhotoImage(image8)
        button2 = tk.Button(image=self.photo_image8, command=self.on_click_8)
        button2.pack()
        button2.place(height=47, width=400, x=100, y=197)

        image9 = Image.open('Pictures//scale_9.jpg')
        self.photo_image9 = ImageTk.PhotoImage(image9)
        button2 = tk.Button(image=self.photo_image9, command=self.on_click_9)
        button2.pack()
        button2.place(height=47, width=400, x=100, y=244)

        image10 = Image.open('Pictures//scale_10.jpg')
        self.photo_image10 = ImageTk.PhotoImage(image10)
        button2 = tk.Button(image=self.photo_image10, command=self.on_click_10)
        button2.pack()
        button2.place(height=50, width=400, x=100, y=291)

        say("Effort")


    def on_click_0(self):
        print("image clicked")
        s.effortClicked = True
        s.effort = 0
        s.screen.switch_frame(BlankPage)

    def on_click_1(self):
        print("image clicked")
        s.effortClicked = True
        s.effort = 1
        s.screen.switch_frame(BlankPage)

    def on_click_2(self):
        print("image clicked")
        s.effortClicked = True
        s.effort = 2
        s.screen.switch_frame(BlankPage)

    def on_click_3(self):
        print("image clicked")
        s.effortClicked = True
        s.effort = 3
        s.screen.switch_frame(BlankPage)

    def on_click_4(self):
        print("image clicked")
        s.effortClicked = True
        s.effort = 4
        s.screen.switch_frame(BlankPage)

    def on_click_5(self):
        print("image clicked")
        s.effortClicked = True
        s.effort = 5
        s.screen.switch_frame(BlankPage)

    def on_click_6(self):
        print("image clicked")
        s.effortClicked = True
        s.effort = 6
        s.screen.switch_frame(BlankPage)

    def on_click_7(self):
        print("image clicked")
        s.effortClicked = True
        s.effort = 7
        s.screen.switch_frame(BlankPage)

    def on_click_8(self):
        print("image clicked")
        s.effortClicked = True
        s.effort = 8
        s.screen.switch_frame(BlankPage)

    def on_click_9(self):
        print("image clicked")
        s.effortClicked = True
        s.effort = 9
        s.screen.switch_frame(BlankPage)

    def on_click_10(self):
        print("image clicked")
        s.effortClicked = True
        s.effort = 10
        s.screen.switch_frame(BlankPage)

class ExercisePage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        image = Image.open('Pictures//exercise.jpg')
        self.photo_image = ImageTk.PhotoImage(image) #self. - for keeping the photo in memory so it will be shown
        tk.Label(self, image = self.photo_image).pack()

class StartPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        image1 = Image.open('Pictures//Start.jpg')
        self.photo_image1 = ImageTk.PhotoImage(image1)
        self.background_label = tk.Label(image=self.photo_image1)
        self.background_label.pack()

        image2 = Image.open('Pictures//StartButton.jpg')
        self.photo_image2 = ImageTk.PhotoImage(image2)
        button2= tk.Button(self.background_label, image=self.photo_image2, command=self.on_click)
        button2.pack()
        button2.place(height=480, width=480, x=265, y=100)

    def on_click(self):
        print("image clicked")
        s.waved = True
        s.screen.switch_frame(BlankPage)


class TryAgainPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        image1 = Image.open('Pictures//tryagain.jpg')
        self.photo_image1 = ImageTk.PhotoImage(image1)
        self.background_label = tk.Label(image=self.photo_image1)
        self.background_label.pack()

        image2 = Image.open('Pictures//tryagainright2.jpg')
        self.photo_image2 = ImageTk.PhotoImage(image2)
        button2= tk.Button(image=self.photo_image2, command=self.on_click_right)
        button2.pack()
        button2.place(height=350, width=350, x=535, y=155)

        image3 = Image.open('Pictures//tryagainleft2.jpg')
        self.photo_image3 = ImageTk.PhotoImage(image3)
        button2= tk.Button(image=self.photo_image3, command=self.on_click_left)
        button2.pack()
        button2.place(height=350, width=360, x=135, y=155)


    def on_click_right(self):
        print("image clicked")
        s.waved = True
        s.screen.switch_frame(BlankPage)

    def on_click_left(self):
        print("image clicked")
        s.clickedTryAgain=True
        s.screen.switch_frame(BlankPage)

class BlankPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        image = Image.open('Pictures//Background.jpg')
        self.photo_image = ImageTk.PhotoImage(image)
        tk.Label(self, image=self.photo_image).pack()

class Very_good(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        image = Image.open('Pictures//verygood.jpg')
        self.photo_image = ImageTk.PhotoImage(image)
        tk.Label(self, image=self.photo_image).pack()
        say('very_good')

class Excellent(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        image = Image.open('Pictures//excellent.jpg')
        self.photo_image = ImageTk.PhotoImage(image)
        tk.Label(self, image=self.photo_image).pack()
        say('excellent')

class Well_done(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        image = Image.open('Pictures//welldone.jpg')
        self.photo_image = ImageTk.PhotoImage(image)
        tk.Label(self, image=self.photo_image).pack()
        say('well_done')


class AlmostExcellent(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        image = Image.open('Pictures//almostexcellent.jpg')
        self.photo_image = ImageTk.PhotoImage(image)
        tk.Label(self, image=self.photo_image).pack()
        say('almostexcellent')


class Fail(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        image = Image.open('Pictures//fail.jpg')
        self.photo_image = ImageTk.PhotoImage(image)
        tk.Label(self, image=self.photo_image).pack()
        say('fail')

class GoodbyePage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        image = Image.open('Pictures//goodbye.jpg')
        self.photo_image = ImageTk.PhotoImage(image)
        tk.Label(self, image=self.photo_image).pack()


class CalibrationPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        image = Image.open('Pictures//calibration.jpg')
        self.photo_image = ImageTk.PhotoImage(image)
        tk.Label(self, image=self.photo_image).pack()
        say('calibration')


class DemoPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        image = Image.open('Pictures//demo.jpg')
        self.photo_image = ImageTk.PhotoImage(image)
        tk.Label(self, image=self.photo_image).pack()

######################################### Counting Pages #############################################################
class OnePage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        image = Image.open('Pictures//1.jpg')
        self.photo_image = ImageTk.PhotoImage(image)
        tk.Label(self, image=self.photo_image).pack()


class TwoPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        image = Image.open('Pictures//2.jpg')
        self.photo_image = ImageTk.PhotoImage(image)
        tk.Label(self, image=self.photo_image).pack()


class ThreePage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        image = Image.open('Pictures//3.jpg')
        self.photo_image = ImageTk.PhotoImage(image)
        tk.Label(self, image=self.photo_image).pack()


class FourPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        image = Image.open('Pictures//4.jpg')
        self.photo_image = ImageTk.PhotoImage(image)
        tk.Label(self, image=self.photo_image).pack()


class FivePage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        image = Image.open('Pictures//5.jpg')
        self.photo_image = ImageTk.PhotoImage(image)
        tk.Label(self, image=self.photo_image).pack()


class SixPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        image = Image.open('Pictures//6.jpg')
        self.photo_image = ImageTk.PhotoImage(image)
        tk.Label(self, image=self.photo_image).pack()


class SevenPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        image = Image.open('Pictures//7.jpg')
        self.photo_image = ImageTk.PhotoImage(image)
        tk.Label(self, image=self.photo_image).pack()


class EightPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        image = Image.open('Pictures//8.jpg')
        self.photo_image = ImageTk.PhotoImage(image)
        tk.Label(self, image=self.photo_image).pack()


class NinePage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        image = Image.open('Pictures//9.jpg')
        self.photo_image = ImageTk.PhotoImage(image)
        tk.Label(self, image=self.photo_image).pack()


class TenPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        image = Image.open('Pictures//10.jpg')
        self.photo_image = ImageTk.PhotoImage(image)
        tk.Label(self, image=self.photo_image).pack()



class FullScreenApp(object):
    def __init__(self, master, **kwargs):
        self.master=master
        pad=3
        self._geom='200x200+0+0'
        master.geometry("{0}x{1}+0+0".format(
            master.winfo_screenwidth()-pad, master.winfo_screenheight()-pad))
        master.bind('<Escape>',self.toggle_geom)

    def toggle_geom(self,event):
        geom=self.master.winfo_geometry()
        print(geom,self._geom)
        self.master.geometry(self._geom)
        self._geom=geom


if __name__ == "__main__":
    s.audio_path = 'audio files/Hebrew/Female/'
    s.screen = Screen()
    s.screen.switch_frame(Well_done)
    app = FullScreenApp(s.screen)
    s.screen.mainloop()


