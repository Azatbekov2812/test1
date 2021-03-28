import cv2
import matplotlib.pyplot as plt
import numpy
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render

# Create your views here.
from monetka import settings


def get_image_data(request):
    if request.method != "POST":
        return render(request, 'coin/home.html')

    if request.method == 'POST':
        file = request.FILES['file']
        fs = FileSystemStorage()
        filename = fs.save(file.name, file)
        circles, total_amount = calculate_amount(filename)
        print(settings.MEDIA_URL)
        image = cv2.imread('media/' + filename, 1)

        h, w, c = image.shape

        avg_color_per_row = numpy.average(image, axis=0)
        avg_color = numpy.average(avg_color_per_row, axis=0)

        data = {
            'avg_color': avg_color,
            'height': h,
            'weight': w,
            'count': len(circles[0]),
            'total_amount': total_amount
        }

        return render(request, 'coin/home.html', data)


def detect_coins(filename):
    iamge = cv2.imread('media/' + filename, 1)
    # gray
    gray = cv2.cvtColor(iamge, cv2.COLOR_BGR2GRAY)

    img = cv2.medianBlur(gray, 7)

    circles = cv2.HoughCircles(
        img,  # source image
        cv2.HOUGH_GRADIENT,  # type of detection
        1,
        50,
        param1=100,
        param2=50,
        minRadius=10,  # minimal radius
        maxRadius=380,  # max radius
    )

    image_copy = iamge.copy()

    coins_detected = None
    for detected_circle in circles[0]:
        x_coor, y_coor, detected_radius = detected_circle
        coins_detected = cv2.circle(
            image_copy,
            (int(x_coor), int(y_coor)),
            int(detected_radius),
            (0, 255, 0),
            4,
        )
    plt.imshow(coins_detected, cmap='gray')
    plt.show()
    cv2.imwrite("media/r" + filename, coins_detected)

    return circles


def calculate_amount(filename):
    koruny = {
        "1 RU": {
            "value": 1,
            "radius": 20,
            "ratio": 1,
            "count": 0,
        },
        "10 RU": {
            "value": 10,
            "radius": 21.5,
            "ratio": 1.075,
            "count": 0,
        },
        "2 RU": {
            "value": 2,
            "radius": 23,
            "ratio": 1.15,
            "count": 0,
        },
        "5 RU": {
            "value": 5,
            "radius": 24.5,
            "ratio": 1.225,
            "count": 0,
        }
    }

    circles = detect_coins(filename)
    radius = []
    coordinates = []

    for detected_circle in circles[0]:
        x_coor, y_coor, detected_radius = detected_circle
        radius.append(detected_radius)
        coordinates.append([x_coor, y_coor])

    smallest = min(radius)
    tolerance = 0.0375
    total_amount = 0

    coins_circled = cv2.imread('media/r' + filename, 1)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for coin in circles[0]:
        ratio_to_check = coin[2] / smallest
        coor_x = coin[0]
        coor_y = coin[1]
        for koruna in koruny:
            value = koruny[koruna]['value']
            if abs(ratio_to_check - koruny[koruna]['ratio']) <= tolerance:
                koruny[koruna]['count'] += 1
                total_amount += koruny[koruna]['value']
                cv2.putText(coins_circled, str(value), (int(coor_x), int(coor_y)), font, 1,
                            (0, 0, 0), 4)

    print(f" Общая сумма монет {total_amount} RU")
    cv2.imwrite("media/r2" + filename, coins_circled)
    return circles, total_amount
