import wx
import threading
import cv2 as cv
import qr
import os
import time
from PIL import Image

overlay = False
#import pyrealsense2 as rs

#baseRvec = [0,0,0]
#baseTvec = [0,0,0]

class CameraThread(threading.Thread):
    def __init__(self, input_source, cmtx, dist, frame_callback, update_callback, main_frame):
        super().__init__()
        self.input_source = input_source
        self.cmtx = cmtx
        self.dist = dist
        self.frame_callback = frame_callback
        self.update_callback = update_callback
        self.main_frame = main_frame  
        self.running = True
        self.current_frame = None

    def run(self):
        if isinstance(self.input_source, int):
            cap = cv.VideoCapture(self.input_source)
        elif isinstance(self.input_source, str):
            cap = cv.VideoCapture(self.input_source)
        else:
            raise ValueError("Invalid input for video ")

        while self.running:
            #print("Camera thread running...")
            ret, frame = cap.read()
            #if ret == False: break
            if ret:
                processed_frame = qr.process_frame(frame, self.cmtx, self.dist, self.update_callback , zoom_factor=self.main_frame.camera_panel.zoom_factor)
                self.current_frame = frame
                wx.CallAfter(self.frame_callback, processed_frame)
                time.sleep(0.05)
            else:
                break
        print("Camera thread loop exited.")
        cap.release()

    def stop(self):
        print("Stopping camera thread...")
        self.running = False

class CameraPanel(wx.Panel):
    def __init__(self, parent, main_frame, size=(800, 600)):
        super().__init__(parent, size=size)
        self.main_frame = main_frame  
        self.camera_bitmap = wx.StaticBitmap(self, size=size)
        self.zoom_factor = 1.0  
        # Initialize zoom factor

        # Create zoom buttons
        zoom_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.zoom_out_button = wx.Button(self, label='-')
        self.zoom_in_button = wx.Button(self, label='+')
        zoom_sizer.Add(self.zoom_out_button, 0, wx.ALL, 5)
        zoom_sizer.Add(self.zoom_in_button, 0, wx.ALL, 5)

        # Bind button events
        self.zoom_out_button.Bind(wx.EVT_BUTTON, self.on_zoom_out)
        self.zoom_in_button.Bind(wx.EVT_BUTTON, self.on_zoom_in)

        # Add zoom buttons to the layout
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(self.camera_bitmap, 1, wx.EXPAND)
        main_sizer.Add(zoom_sizer, 0, wx.ALIGN_CENTER)
        self.SetSizer(main_sizer)

    def update_frame(self, frame):
        height, width = frame.shape[:2]
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        
        if self.main_frame.return_mode_active:
            selected_image = self.main_frame.image_selector.GetValue()
            if selected_image and qr.returnToLoc(self.main_frame):
                overlay_image_path = os.path.join('savedImages', 'images', selected_image)
                if os.path.exists(overlay_image_path):
                    overlay_image = cv.imread(overlay_image_path)
                    overlay_image = cv.resize(overlay_image, (width, height))
                    frame = cv.addWeighted(frame, 1, overlay_image, 0.5, 0)
        
        bitmap = wx.Bitmap.FromBuffer(width, height, frame)
        self.camera_bitmap.SetBitmap(bitmap)
                    
    # Decrease zoom factor
    def on_zoom_out(self, event):
        self.zoom_factor = max(self.zoom_factor - 0.1, 1.0)  
        print(f"Zoom factor: {self.zoom_factor}")

    # Increase zoom factor
    def on_zoom_in(self, event):
        self.zoom_factor = min(self.zoom_factor + 0.1, 5.0)  
        print(f"Zoom factor: {self.zoom_factor}")

class HelloFrame(wx.Frame):
    def __init__(self, *args, **kw):
        super(HelloFrame, self).__init__(*args, **kw, size=(800, 600))
        self.SetTitle('QR Code Tracker')

        pnl = wx.Panel(self)
        self.Bind(wx.EVT_CLOSE, self.on_close)
        # Main sizer for the entire panel
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # Camera Panel
        self.camera_panel = CameraPanel(pnl, self, size=(800, 600))
        main_sizer.Add(self.camera_panel, 1, wx.EXPAND | wx.ALL, 5)

        # Buttons at the bottom of the screen
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.save_button = wx.Button(pnl, label="Save")
        self.return_button = wx.ToggleButton(pnl, label="Return")

        self.return_mode_active = False

        #Button to toggle the return image overlay
        self.overlay_button = wx.ToggleButton(pnl, label="Overlay")

        # Add buttons to the button sizer
        button_sizer.Add(self.save_button, 0, wx.EXPAND | wx.ALL, 5)
        button_sizer.Add(self.return_button, 0, wx.EXPAND | wx.ALL, 5)
        button_sizer.Add(self.overlay_button, 0, wx.EXPAND | wx.ALL, 5)

        
        # Add the button sizer to the main sizer
        main_sizer.Add(button_sizer, 0, wx.ALIGN_CENTER | wx.ALL, 5)

        pnl.SetSizer(main_sizer)
        
        # Add StaticText widgets for displaying coordinates
        self.tvec_text = wx.StaticText(pnl, label="Translation Vector: [0, 0, 0]")
        self.rvec_text = wx.StaticText(pnl, label="Rotation Vector: [0, 0, 0]")
        
        # Add the StaticText widgets to the main sizer
        main_sizer.Add(self.tvec_text, 0, wx.ALIGN_CENTER | wx.ALL, 5)
        main_sizer.Add(self.rvec_text, 0, wx.ALIGN_CENTER | wx.ALL, 5)

        # Start the camera thread
        cmtx, dist = qr.read_camera_parameters()
        input_source = self.ShowOptionDialog()
        self.camera_thread = CameraThread(input_source, cmtx, dist, self.camera_panel.update_frame, self.update_coordinates, self)
        self.camera_thread.start()

        # Bind event handlers for the buttons
        self.save_button.Bind(wx.EVT_BUTTON, self.OnSaveClick)
        self.return_button.Bind(wx.EVT_TOGGLEBUTTON, self.OnToggle)
        self.overlay_button.Bind(wx.EVT_TOGGLEBUTTON, self.OnToggleReturnOverlay)

        self.Bind(wx.EVT_CLOSE, self.on_close)

        # Maximize the frame to open in full screen
        self.Maximize(True)

        # Add a text region to display the percentage on return feture on
        self.percentage_display = wx.StaticText(pnl, label="Percentage: 0%")
        main_sizer.Add(self.percentage_display, 0, wx.ALIGN_CENTER | wx.ALL, 5)

        # Add a StaticText for the status indicator
        self.status_indicator = wx.StaticText(pnl, label="Status: Not Matched")
        main_sizer.Add(self.status_indicator, 0, wx.ALIGN_CENTER | wx.ALL, 5)

        # ComboBox for selecting images
        self.image_selector = wx.ComboBox(pnl, choices=[], style=wx.CB_READONLY)
        main_sizer.Add(self.image_selector, 0, wx.EXPAND | wx.ALL, 5)

        # Bind the EVT_COMBOBOX event
        self.image_selector.Bind(wx.EVT_COMBOBOX, self.OnImageSelected)

        # create a menu bar
        self.makeMenuBar()

        # and a status bar
        #self.CreateStatusBar()
        #self.SetStatusText("Welcome to wxPython!")
    def update_coordinates(self, tvec, rvec):
        # Format the vectors for display
        tvec_str = f"[{tvec[0][0]:.2f}, {tvec[1][0]:.2f}, {tvec[2][0]:.2f}]"
        rvec_str = f"[{rvec[0][0]:.2f}, {rvec[1][0]:.2f}, {rvec[2][0]:.2f}]"

        # Update the StaticText widgets
        self.tvec_text.SetLabel(f"Translation Vector: {tvec_str}")
        self.rvec_text.SetLabel(f"Rotation Vector: {rvec_str}")

    def populate_image_selector(self):
        saved_images_dir = r'savedImages/images'
        image_files = [f for f in os.listdir(saved_images_dir) if os.path.isfile(os.path.join(saved_images_dir, f))]
        self.image_selector.Clear()
        self.image_selector.AppendItems(image_files)

    def OnImageSelected(self, event):
        selected_image = self.image_selector.GetValue()
        if selected_image:
            print(f"Returning to location associated with {selected_image}")
            qr.saved_tvec, qr.saved_rvec = qr.getInitialPoints(selected_image)
            matched, percentage = qr.returnToLoc(self)
            self.update_status(matched, percentage)
        else:
            print("No image selected")

    def update_status(self, matched, percentage=0):
        if matched:
            self.status_indicator.SetLabel("Status: Matched")
            self.status_indicator.SetForegroundColour(wx.Colour(0, 255, 0))  
            print("Status: Matched")
        else:
            self.status_indicator.SetLabel("Status: Not Matched")
            self.status_indicator.SetForegroundColour(wx.Colour(255, 0, 0)) 
            print("Status: Not Matched")
        
        print(f"Percentage: {percentage}%")
        self.percentage_display.SetLabel(f"Percentage: {percentage}%")

    def on_close(self, event):
        print("Closing application...")
        self.camera_thread.stop()
        self.camera_thread.join()
        print("Camera thread stopped.")
        self.Destroy()
        print("Application closed.")   

    def makeMenuBar(self):
        """
        A menu bar is composed of menus, which are composed of menu items.
        This method builds a set of menus and binds handlers to be called
        when the menu item is selected.
        """

        # Make a file menu with Hello and Exit items
        fileMenu = wx.Menu()
        # The "\t..." syntax defines an accelerator key that also triggers
        # the same event
        helloItem = fileMenu.Append(-1, "&Hello...\tCtrl-H",
                "Help string shown in status bar for this menu item")
        fileMenu.AppendSeparator()
        # When using a stock ID we don't need to specify the menu item's
        # label
        exitItem = fileMenu.Append(wx.ID_EXIT)
        
        #new menu item for toggling the return overlay
        #self.returnOverlayItem = fileMenu.Append(wx.ID_ANY, "Return Overlay\tCtrl-R", "Toggle return overlay", kind=wx.ITEM_CHECK)

        # Now a help menu for the about item
        helpMenu = wx.Menu()
        aboutItem = helpMenu.Append(wx.ID_ABOUT)

        # Make the menu bar and add the two menus to it. The '&' defines
        # that the next letter is the "mnemonic" for the menu item. On the
        # platforms that support it those letters are underlined and can be
        # triggered from the keyboard.
        menuBar = wx.MenuBar()
        menuBar.Append(fileMenu, "&File")
        menuBar.Append(helpMenu, "&Help")
        #menuBar.Append(helpMenu, "&Return")

        # Give the menu bar to the frame
        self.SetMenuBar(menuBar)

        # Finally, associate a handler function with the EVT_MENU event for
        # each of the menu items. That means that when that menu item is
        # activated then the associated handler function will be called.
        self.Bind(wx.EVT_MENU, self.OnHello, helloItem)
        self.Bind(wx.EVT_MENU, self.OnExit,  exitItem)
        self.Bind(wx.EVT_MENU, self.OnAbout, aboutItem)
        #self.Bind(wx.EVT_MENU, self.OnToggleReturnOverlay, self.returnOverlayItem)

    def OnToggleReturnOverlay(self, event):
        self.return_mode_active = not self.return_mode_active
        if self.return_mode_active:
            print("Return overlay activated")
            #self.returnOverlayItem.Check(True)
            event.GetEventObject().SetLabel("Overlay: on")
        else:
            print("Return overlay deactivated")
            #self.returnOverlayItem.Check(False)
            event.GetEventObject().SetLabel("Overalay: off")
        self.camera_panel.Refresh()


    def OnExit(self, event):
        """Close the frame, terminating the application."""
        self.Close(True)

    def OnHello(self, event):
        """Say hello to the user."""
        wx.MessageBox("Hello again from wxPython")

    def OnAbout(self, event):
        """Display an About Dialog"""
        wx.MessageBox("This is a wxPython Hello World sample",
                      "About Hello World 2",
                      wx.OK|wx.ICON_INFORMATION)
        
    def OnToggle(self,event): 
        #state = event.GetEventObject().GetValue() 

        self.return_mode_active = event.GetEventObject().GetValue()

        if self.return_mode_active:
            print("Return mode activated")
            self.populate_image_selector()
            event.GetEventObject().SetLabel("Return: on")
        else:
            self.update_status(False)
            print("Return mode deactivated")
            event.GetEventObject().SetLabel("Return: off")

    def OnSaveClick(self, event):
        global saved_rvec, saved_tvec, current_rvec, current_tvec

        current_frame = self.camera_thread.current_frame
        if qr.current_rvec is not None and qr.current_tvec is not None and current_frame is not None:
            qr.saveImageNLoc(qr.current_rvec, qr.current_tvec, current_frame)
            print("Saved location:\n", qr.current_rvec, qr.current_tvec)
        else:
            print("No QR code to save ")

        btn = event.GetEventObject().GetLabel() 
        print("Label of pressed button = ",btn )

    '''def ShowOptionDialog(self):
        dlg = wx.MessageDialog(self, "Choose an option:", "Options", wx.YES_NO | wx.ICON_QUESTION)

        # Set labels for the buttons
        dlg.SetYesNoLabels("Camera", "Test Video")

        result = dlg.ShowModal()
        if result == wx.ID_YES:
            feedOpt = 0
            print("Option 1 selected")
        elif result == wx.ID_OK:
            feedOpt = 1
            print("Option 2 selected")
        elif result == wx.ID_CANCEL:
            feedOpt = 2
            print("Option 3 selected")
        elif result == wx.ID_NO:
            feedOpt = 'media/test.mp4'
            print("Option 4 selected")

        dlg.Destroy()
        return feedOpt'''
    
    def ShowOptionDialog(self):
        # Create a custom dialog
        dlg = wx.Dialog(self, title="Options", size=(300, 200))

        # Create a sizer for layout
        sizer = wx.BoxSizer(wx.VERTICAL)

        # Add a message
        message = wx.StaticText(dlg, label="Choose an option:")
        sizer.Add(message, 0, wx.ALL | wx.CENTER, 10)

        # Create buttons for options
        camera_btn = wx.Button(dlg, id=wx.ID_ANY, label="Cam @ in 0")
        option3_btn = wx.Button(dlg, id=wx.ID_ANY, label="Cam @ in 1")
        option4_btn = wx.Button(dlg, id=wx.ID_ANY, label="Cam @ in 2")
        test_video_btn = wx.Button(dlg, id=wx.ID_ANY, label="Test Video")

        # Add buttons to sizer
        sizer.Add(camera_btn, 0, wx.ALL | wx.CENTER, 5)
        sizer.Add(option3_btn, 0, wx.ALL | wx.CENTER, 5)
        sizer.Add(option4_btn, 0, wx.ALL | wx.CENTER, 5)
        sizer.Add(test_video_btn, 0, wx.ALL | wx.CENTER, 5)

        # Bind events for buttons
        camera_btn.Bind(wx.EVT_BUTTON, lambda event: dlg.EndModal(0))
        option3_btn.Bind(wx.EVT_BUTTON, lambda event: dlg.EndModal(1))
        option4_btn.Bind(wx.EVT_BUTTON, lambda event: dlg.EndModal(2))
        test_video_btn.Bind(wx.EVT_BUTTON, lambda event: dlg.EndModal(3))
        # Set sizer and layout
        dlg.SetSizer(sizer)
        dlg.Layout()

        # Show the dialog and get the result
        result = dlg.ShowModal()
        dlg.Destroy()

        # Handle the result
        if result == 0:
            feedOpt = 0
            print("Camera selected")
        elif result == 1:
            feedOpt = 1
            print("Option 3 selected")
        elif result == 2:
            feedOpt = 2
            print("Option 4 selected")
        elif result == 3:
            feedOpt = 'media/test.mp4'
            print("Test Video selected")
        else:
            feedOpt = None

        return feedOpt

class MyApp(wx.App):
    def OnInit(self):
        frame = HelloFrame(None, "Option Dialog")
        frame.Show(True)
        return True

if __name__ == '__main__':
    app = wx.App(False)
    frm = HelloFrame(None, title='QR Code Tracker')
    frm.Show()
    app.MainLoop()