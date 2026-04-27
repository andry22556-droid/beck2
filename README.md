# Разработка графического интерфейса для кейса "Сбор аналитических данных блога"
### :hammer_and_wrench: Languages and Tools :

<div align="center">
  <img src="https://raw.githubusercontent.com/devicons/devicon/1119b9f84c0290e0f0b38982099a2bd027a48bf1/icons/python/python-original.svg" height="40" width="40">
  <img src="https://raw.githubusercontent.com/devicons/devicon/1119b9f84c0290e0f0b38982099a2bd027a48bf1/icons/fastapi/fastapi-original.svg" height="40" width="40">
  <img src="https://raw.githubusercontent.com/devicons/devicon/1119b9f84c0290e0f0b38982099a2bd027a48bf1/icons/react/react-original.svg" height="40" width="40">
</div>

## :crystal_ball: Front-End
Загрузка зависимостей
```
cd frontend
npm install --legacy-peer-deps
```
Запуск frontend приложения
```
npm start
```
Сервер будет работать по ссылке 127.0.0.1:3000
## :hammer: Back-End
Создание виртуального окружения:
```
python -m venv backend/venv
backend\venv\Scripts\Activate.ps1
```
Установка необходимых пакетов:

Будьте очень осторожны, зависимости очень чувствительные
```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r ./backend/requirements.txt
```
Если у вас проблемы с Cuda то просто измените немного файл config.py
```
reader = Reader(lang_list=['ru', 'en'], gpu=False)
```

Запуск backend приложения
```
python ./backend/main.py
```
Сервер будет работать по ссылке 127.0.0.1:3000
## :moyai: Описание приложения

В данном веб приложении релизовано комплексное решение включающее предобработку изображений и последующее выявление необходимой информации, необходимой для оценки показателей успешности блогов.

Данное решение предпологает то, что на внешних сервисах происходит сбор данных (изображений), после чего, сотрудник загружает собранные данные В ВИДЕ, и получает результат как табличный вывод в интерфейсе, а также файловый вывод в виде csv-файла.

С демонстрацией решения можно ознакомиться [здесь](https://disk.yandex.ru/d/CXjepKwdMqHquw).
