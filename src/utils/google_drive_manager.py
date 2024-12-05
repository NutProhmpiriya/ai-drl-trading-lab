import os
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
import pickle
from typing import Optional
import io

class GoogleDriveManager:
    def __init__(self):
        self.SCOPES = ['https://www.googleapis.com/auth/drive.file']
        self.creds = None
        self.service = None
        self.base_folders = {
            'csv_files': None,
            'models': None
        }
        self._authenticate()
        self._setup_base_folders()

    def _authenticate(self):
        """Authenticate with Google Drive."""
        creds_path = os.path.join('credentials', 'credentials.json')
        token_path = os.path.join('credentials', 'token.pickle')

        if os.path.exists(token_path):
            with open(token_path, 'rb') as token:
                self.creds = pickle.load(token)

        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(creds_path, self.SCOPES)
                self.creds = flow.run_local_server(port=0)

            with open(token_path, 'wb') as token:
                pickle.dump(self.creds, token)

        self.service = build('drive', 'v3', credentials=self.creds)
        print("Successfully authenticated with Google Drive")

    def _setup_base_folders(self):
        """Create or get base folders for CSV files and models."""
        for folder_name in self.base_folders.keys():
            folder_id = self._find_folder(folder_name)
            if not folder_id:
                folder_id = self._create_folder(folder_name)
                print(f"Created new folder: {folder_name}")
            self.base_folders[folder_name] = folder_id
            print(f"Using folder '{folder_name}' with ID: {folder_id}")

    def _find_folder(self, folder_name: str) -> Optional[str]:
        """Find folder by name and return its ID."""
        query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        results = self.service.files().list(q=query, spaces='drive', fields='files(id)').execute()
        items = results.get('files', [])
        return items[0]['id'] if items else None

    def _create_folder(self, folder_name: str) -> str:
        """Create a folder and return its ID."""
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        file = self.service.files().create(body=file_metadata, fields='id').execute()
        return file.get('id')

    def upload_file(self, file_path: str, file_type: str) -> Optional[str]:
        """Upload a file to the appropriate folder based on its type."""
        if file_type not in self.base_folders:
            raise ValueError(f"Invalid file type. Must be one of: {list(self.base_folders.keys())}")

        folder_id = self.base_folders[file_type]
        file_name = os.path.basename(file_path)

        try:
            file_metadata = {
                'name': file_name,
                'parents': [folder_id]
            }
            media = MediaFileUpload(file_path, resumable=True)
            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            print(f"Successfully uploaded {file_name} to {file_type} folder")
            return file.get('id')
        except Exception as e:
            print(f"Error uploading file: {e}")
            return None

    def download_file(self, file_id: str, destination_path: str) -> bool:
        """Download a file from Google Drive."""
        try:
            request = self.service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False

            while not done:
                status, done = downloader.next_chunk()
                print(f"Download {int(status.progress() * 100)}%")

            fh.seek(0)
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            with open(destination_path, 'wb') as f:
                f.write(fh.read())

            print(f"Successfully downloaded file to {destination_path}")
            return True
        except Exception as e:
            print(f"Error downloading file: {e}")
            return False

    def list_files(self, file_type: str) -> list:
        """List all files in the specified folder type."""
        if file_type not in self.base_folders:
            raise ValueError(f"Invalid file type. Must be one of: {list(self.base_folders.keys())}")

        folder_id = self.base_folders[file_type]
        query = f"'{folder_id}' in parents and trashed=false"
        
        try:
            results = self.service.files().list(
                q=query,
                spaces='drive',
                fields='files(id, name, createdTime, modifiedTime)'
            ).execute()
            
            files = results.get('files', [])
            if not files:
                print(f"No files found in {file_type} folder")
            else:
                print(f"\nFiles in {file_type} folder:")
                for file in files:
                    print(f"Name: {file['name']}, ID: {file['id']}")
            return files
        except Exception as e:
            print(f"Error listing files: {e}")
            return []

    def delete_file(self, file_id: str) -> None:
        """Delete a file from Google Drive.
        
        Args:
            file_id: ID of the file to delete
        """
        try:
            self.service.files().delete(fileId=file_id).execute()
            print(f"Successfully deleted file with ID: {file_id}")
        except Exception as e:
            print(f"Error deleting file: {str(e)}")
