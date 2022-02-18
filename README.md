В репозитории представлен классификатор мужских и женских голосов на базе фреймворка PyTorch.

Код находится в ветке master.


## Данные

Использовался датасет Libritts (train-clean-100), который содержит 33236 записей мужских и женских голосов с частотой дискретизации 24000 Гц.


## Модель

Аудиоданные были превращены в мел-спектрограммы, после чего они обрабатывались уже как картинки.
Для классификации использовалась Inception_v3, так как эта модель обучается очень быстро, при этом известна хорошей точностью в задачах компьютерного зрения. Для ускорения обучения была взята предобученная на ImageNet Inception_v3. Такой подход оправдан в статье https://arxiv.org/pdf/2007.11154.pdf , авторы которой пробовали применять различные архитектуры, предобученные на ImageNet, для классификации аудио и достигли хорошей точности.


Для запуска нуно запустить скрипт main.py, также понадобится файл speakers.tsv

## Результаты

Пока что модель обучена только на малом количестве эпох

Модель обучалась в Google Colab на GPU.


## Что можно сделать?

- Аугментрировать данные, например, добавить шум в аудиодорожки или добавить паузы в запись
- Побробовать другие модели: например, CNN с Dilated Convolutions могла бы хорошо подойти для аудиоданных
