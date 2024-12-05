from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
import os
import pickle
import io

class GoogleDriveConnector:
    def __init__(self):
        self.SCOPES = ['https://www.googleapis.com/auth/drive']
        self.creds = None
        self.service = None
        
    def authenticate(self, credentials_path='credentials.json', token_path='token.pickle'):
        """Authenticate with Google Drive"""
        if os.path.exists(token_path):
            with open(token_path, 'rb') as token:
                self.creds = pickle.load(token)
                
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(credentials_path, self.SCOPES)
                self.creds = flow.run_local_server(port=0)
            
            with open(token_path, 'wb') as token:
                pickle.dump(self.creds, token)
        
        self.service = build('drive', 'v3', credentials=self.creds)
        
    def list_files(self, folder_id=None, file_type=None):
        """List files in Google Drive or specific folder"""
        query = []
        if folder_id:
            query.append(f"'{folder_id}' in parents")
        if file_type:
            query.append(f"mimeType='{file_type}'")
            
        query_string = ' and '.join(query) if query else ''
        
        results = self.service.files().list(
            q=query_string,
            pageSize=100,
            fields="nextPageToken, files(id, name, mimeType)"
        ).execute()
        
        return results.get('files', [])
    
    def create_folder(self, folder_name, parent_id=None):
        """Create a new folder in Google Drive"""
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        
        if parent_id:
            file_metadata['parents'] = [parent_id]
            
        file = self.service.files().create(
            body=file_metadata,
            fields='id'
        ).execute()
        
        return file.get('id')
    
    def upload_file(self, file_path, folder_id=None):
        """Upload a file to Google Drive"""
        file_metadata = {'name': os.path.basename(file_path)}
        if folder_id:
            file_metadata['parents'] = [folder_id]
            
        media = MediaFileUpload(
            file_path,
            resumable=True
        )
        
        file = self.service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        
        return file.get('id')
    
    def download_file(self, file_id, output_path):
        """Download a file from Google Drive"""
        request = self.service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        
        while done is False:
            status, done = downloader.next_chunk()
            
        fh.seek(0)
        with open(output_path, 'wb') as f:
            f.write(fh.read())
            
    def delete_file(self, file_id):
        """Delete a file from Google Drive"""
        self.service.files().delete(fileId=file_id).execute()
