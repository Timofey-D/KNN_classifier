# KNN_classifier
The project is a homework by computer vision. It is required to implement a program allowing to differ Hebrew symbols.

The program was implemented 2 students from Sami Shamoon College of Engineering. \

*Description*



Description 
The Classifier for Handwritten Hebrew Letters program, using the k-Nearest Neighbor algorithm, classifies images of letters from the proposed database consisting of handwritten Hebrew letters. To do this, we do preprocessing of files from the database, namely, we convert images to shades of gray (greyscale), add white margins to the images so that its size is square, after which we convert the image to a single size (32,32). The next step is to divide the modified database into several sets, namely a set for training in the amount of 80 percent of the database, a validation set of 10 percent of the database and a test set of 10 percent of the database.
At the training stage, we train the classified k-NN on different values of k, after which we choose the best value of k that provides the greatest accuracy. After we have found the optimal value of k, we evaluate the results of k-NN on the test set and write a file with the results (result.txt ) and the matrix "Confu

(Программа Classifier for Handwritten Hebrew Letters, использюя алгоритм k-Nearest Neighbor, классификацирует изображения букв из предложенной базы данных, состоящей из рукописных букв иврита. Для этого мы делаем предварительную обработку файлов из базы данных, а именно преобразуем изображения в оттенки серого (greyscale), добавляем белые отступы к изображениям, чтобы его размер был квадратным, после чего преобразуем изображение в единый размер (32,32). Следующий этам это разделение измененной базы данных нанесколько наборов, а именно набор для обучения в размере 80 процентов от базы данных, набор для валидации 10 процентов от БД и набор для тестов в размере 10 процентов от БД.
На этапе обучениея мы обучаем классифицированную k-NN на разных значениях k, после чего выбираем наилучшее значение k, обеспечивающего наибольшую точность. После того как мы нашли оптимальное значение k, мы оцениваем результаты k-NN на тестовом наборе и запимываем файл с результатами (result.txt) и матрицу "Confusion matrix" в формате csv.)

Environment

Compile and run Classifier for Handwritten Hebrew Letters program written under Linux with python.
First of all, you need to install or update python using the commands:
- apt update
- apt install python3
Next, you need to install the necessary libraries to work with the database, namely os, sys, sklearn, preprocessing, learning, cv2, numpy via the command:
- pip3 install [library name]

(В первую очередь необходимо установить или обновить python с помощью команд:
- apt update
- apt install python3
Далее необходимо установить необходимые библиотеки для работы с базой данных, а именно os, sys, sklearn, preprocessing, learning, cv2, numpy через команду:
- pip3 install [название библиотеки])


!!!How to Run Your Program Provide instructions and examples so users how to run the program. Describe if there are any assumptions on the input. You can also include screenshots to show examples.!!!
