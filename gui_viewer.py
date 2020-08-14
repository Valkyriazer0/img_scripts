import PySimpleGUI as sg
import cv2
from imgprocessing import img, path


def main():
    sg.theme("LightGreen")

    layout = [
        [sg.Text("OpenCV Demo", size=(60, 1), justification="center")],
        [sg.Image(filename="", key="-IMAGE-")],
        [sg.Radio("None", "Radio", True, size=(10, 1))],
        [
            sg.Radio("canny", "Radio", size=(10, 1), key="-CANNY-"),
            sg.Slider(
                (0, 1000),
                300,
                1,
                orientation="h",
                size=(20, 15),
                key="-CANNY SLIDER A-",
            ),
            sg.Slider(
                (0, 1000),
                300,
                1,
                orientation="h",
                size=(20, 15),
                key="-CANNY SLIDER B-",
            ),
        ],
        [
            sg.Radio("blur", "Radio", size=(10, 1), key="-BLUR-"),
            sg.Slider(
                (1, 11),
                1,
                1,
                orientation="h",
                size=(40, 15),
                key="-BLUR SLIDER-",
            ),
        ],
        [sg.Button("Exit", size=(10, 1))],
    ]

    window = sg.Window("OpenCV Integration", layout, resizable=True, location=(800, 400))
    input_path = path.input_file_path_select()[0]
    img = img.load_img(input_path, "color_bgr")

    while True:
        event, values = window.read(timeout=20)
        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        tmp = img

        if values["-CANNY-"]:
            tmp = cv2.Canny(
                tmp, values["-CANNY SLIDER A-"], values["-CANNY SLIDER B-"]
            )
        elif values["-BLUR-"]:
            tmp = cv2.GaussianBlur(tmp, (21, 21), values["-BLUR SLIDER-"])

        imgbytes = cv2.imencode(".png", tmp)[1].tobytes()
        window["-IMAGE-"].update(data=imgbytes)

    window.close()


if __name__ == '__main__':
    main()
