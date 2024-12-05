# Google Drive Service Documentation

ระบบจัดการไฟล์สำหรับ AI DRL Trading Project ที่ใช้ Google Drive เป็นที่เก็บข้อมูล

## โครงสร้างระบบ

```
ai-drl-trading-lab/
├── credentials/
│   ├── credentials.json    # Google Drive API credentials
│   └── token.pickle       # Authentication token
├── src/
│   ├── utils/
│   │   └── google_drive_manager.py  # Google Drive management class
│   └── gdrive_cli.py      # Command line interface
└── docs/
    └── google_drive_service.md  # Documentation
```

## การติดตั้ง

1. ติดตั้ง dependencies:
```bash
pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client
```

2. วาง credentials.json ในโฟลเดอร์ credentials/

## การใช้งาน CLI

### 1. อัพโหลดไฟล์

อัพโหลดไฟล์ CSV:
```bash
python src/gdrive_cli.py upload --type csv_files --file path/to/your/file.csv
```

อัพโหลด Model:
```bash
python src/gdrive_cli.py upload --type models --file path/to/your/model.zip
```

### 2. ดาวน์โหลดไฟล์

```bash
python src/gdrive_cli.py download --file-id FILE_ID --file path/to/save/file
```

### 3. ดูรายการไฟล์

ดูรายการไฟล์ CSV:
```bash
python src/gdrive_cli.py list --type csv_files
```

ดูรายการ Models:
```bash
python src/gdrive_cli.py list --type models
```

## โครงสร้าง Google Drive

ระบบจะสร้างโฟลเดอร์ 2 โฟลเดอร์ใน Google Drive:
1. `csv_files`: สำหรับเก็บข้อมูล Forex (CSV)
   - Columns: high, low, close, open, tick_volume, spread, real_volume

2. `models`: สำหรับเก็บ trained models

## การ Authentication

1. เมื่อรันครั้งแรก ระบบจะเปิด web browser เพื่อขอ authorization
2. หลังจาก authorize แล้ว ระบบจะเก็บ token ไว้ใน credentials/token.pickle
3. การรันครั้งต่อไปจะใช้ token ที่เก็บไว้โดยอัตโนมัติ

## ข้อควรระวัง

1. อย่าลบหรือเปลี่ยนชื่อโฟลเดอร์ `csv_files` และ `models` ใน Google Drive
2. อย่าแชร์ไฟล์ credentials.json และ token.pickle
3. ตรวจสอบว่ามีพื้นที่เพียงพอใน Google Drive ก่อนอัพโหลดไฟล์

## การแก้ปัญหาเบื้องต้น

1. หากมีปัญหาเรื่อง authentication:
   - ลบไฟล์ token.pickle
   - รันคำสั่งใหม่เพื่อ authenticate อีกครั้ง

2. หากอัพโหลด/ดาวน์โหลดไม่สำเร็จ:
   - ตรวจสอบการเชื่อมต่ออินเทอร์เน็ต
   - ตรวจสอบสิทธิ์การเข้าถึงไฟล์
   - ตรวจสอบพื้นที่ว่างใน Google Drive

## การพัฒนาต่อ

หากต้องการพัฒนาเพิ่มเติม สามารถดูโค้ดได้ที่:
1. `src/utils/google_drive_manager.py`: สำหรับแก้ไขการทำงานกับ Google Drive
2. `src/gdrive_cli.py`: สำหรับแก้ไข Command Line Interface
