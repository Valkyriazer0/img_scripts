import wx


class MyFileDropTarget(wx.FileDropTarget):
    def __init__(self, window):
        wx.FileDropTarget.__init__(self)
        self.window = window

    def OnDropFiles(self, x, y, filenames):
        # ファイルパスをテキストフィールドに表示
        for file in filenames:
            self.window.text.SetValue(file)
        return True


class MyFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, title="Drop Target", size=(500, 200))
        p = wx.Panel(self)
        sizer = wx.BoxSizer(wx.HORIZONTAL)

        label = wx.StaticText(p, -1, "File name:")
        self.text = wx.TextCtrl(p, -1, "", size=(400, -1))
        sizer.Add(label, 0, wx.ALL, 5)
        sizer.Add(self.text, 0, wx.ALL, 5)
        p.SetSizer(sizer)

        dt = MyFileDropTarget(self)
        self.SetDropTarget(dt)
        self.Show()


if __name__ == '__main__':
    app = wx.App()
    MyFrame()
    app.MainLoop()
