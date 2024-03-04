import wx
import threading
import cv2 as cv
import qr

#baseRvec = [0,0,0]
#baseTvec = [0,0,0]

class CameraThread(threading.Thread):
    def __init__(self, input_source, cmtx, dist, frame_callback):
        super().__init__()
        self.input_source = input_source
        self.cmtx = cmtx
        self.dist = dist
        self.frame_callback = frame_callback
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
            ret, frame = cap.read()
            #if ret == False: break
            if ret:
                processed_frame = qr.process_frame(frame, self.cmtx, self.dist)
                self.current_frame = frame
                wx.CallAfter(self.frame_callback, processed_frame)
            else:
                break
        cap.release()

    def stop(self):
        self.running = False

class CameraPanel(wx.Panel):
    def __init__(self, parent, size=(800, 600)):
        super().__init__(parent, size=size)
        self.camera_bitmap = wx.StaticBitmap(self, size=size)

    def update_frame(self, frame):
        height, width = frame.shape[:2]
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        bitmap = wx.Bitmap.FromBuffer(width, height, frame)
        self.camera_bitmap.SetBitmap(bitmap)

class HelloFrame(wx.Frame):
    def __init__(self, *args, **kw):
        super(HelloFrame, self).__init__(*args, **kw, size=(800, 600))
        self.SetTitle('QR Code Tracker')

        pnl = wx.Panel(self)

        # Main sizer for the entire panel
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # Camera Panel
        self.camera_panel = CameraPanel(pnl, size=(800, 600))
        main_sizer.Add(self.camera_panel, 1, wx.EXPAND | wx.ALL, 5)

        # Buttons at the bottom of the screen
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.save_button = wx.Button(pnl, label="Save")
        self.return_button = wx.ToggleButton(pnl, label="Return")

        # Add buttons to the button sizer
        button_sizer.Add(self.save_button, 0, wx.EXPAND | wx.ALL, 5)
        button_sizer.Add(self.return_button, 0, wx.EXPAND | wx.ALL, 5)

        
        # Add the button sizer to the main sizer
        main_sizer.Add(button_sizer, 0, wx.ALIGN_CENTER | wx.ALL, 5)

        pnl.SetSizer(main_sizer)

        # Start the camera thread
        cmtx, dist = qr.read_camera_parameters()
        input_source = self.ShowOptionDialog()
        self.camera_thread = CameraThread(input_source, cmtx, dist, self.camera_panel.update_frame)
        self.camera_thread.start()

        # Bind event handlers for the buttons
        self.save_button.Bind(wx.EVT_BUTTON, self.OnSaveClick)
        self.return_button.Bind(wx.EVT_TOGGLEBUTTON, self.OnToggle)

        self.Bind(wx.EVT_CLOSE, self.on_close)

        # Maximize the frame to open in full screen
        self.Maximize(True)

        # create a menu bar
        self.makeMenuBar()

        # and a status bar
        #self.CreateStatusBar()
        #self.SetStatusText("Welcome to wxPython!")

    def on_close(self, event):
        # Stop the camera thread and close the frame
        self.camera_thread.stop()
        self.camera_thread.join()
        self.Destroy()    

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

        # Give the menu bar to the frame
        self.SetMenuBar(menuBar)

        # Finally, associate a handler function with the EVT_MENU event for
        # each of the menu items. That means that when that menu item is
        # activated then the associated handler function will be called.
        self.Bind(wx.EVT_MENU, self.OnHello, helloItem)
        self.Bind(wx.EVT_MENU, self.OnExit,  exitItem)
        self.Bind(wx.EVT_MENU, self.OnAbout, aboutItem)

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
        state = event.GetEventObject().GetValue() 
        if state == True: 
            print( "off" )
            event.GetEventObject().SetLabel("Return?") 
        else: 
            print ("on" )
            event.GetEventObject().SetLabel("Returning")

    def OnSaveClick(self, event):
        global saved_rvec, saved_tvec, current_rvec, current_tvec

        current_frame = self.camera_thread.current_frame

        #qr.saveImageNLoc(qr.current_rvec, qr.current_tvec)
        if qr.current_rvec is not None and qr.current_tvec is not None and current_frame is not None:
            qr.saveImageNLoc(qr.current_rvec, qr.current_tvec, current_frame)
            print("Saved location:\n", qr.current_rvec, qr.current_tvec)
        else:
            print("No QR code to save ")

        btn = event.GetEventObject().GetLabel() 
        print("Label of pressed button = ",btn )

    def ShowOptionDialog(self):
        dlg = wx.MessageDialog(self, "Choose an option:", "Options", wx.YES_NO | wx.ICON_QUESTION)

        # Set custom labels for the buttons
        dlg.SetYesNoLabels("Camera", "Test Video")

        result = dlg.ShowModal()
        if result == wx.ID_YES:
            feedOpt = 1
            print("Option 1 selected")
            # Call the function for Option 1 here
        elif result == wx.ID_NO:
            feedOpt = 'media/test.mp4'
            print("Option 2 selected")
            # Call the function for Option 2 here
        dlg.Destroy()
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