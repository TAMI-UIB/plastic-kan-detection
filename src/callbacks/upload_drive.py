import os
import io
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError

dataset_id = {"sentinel2": "1kQrYXVMxw_MdlP2Mmm69QXwu9yqlg4SP", "marida": "1qQsnXnSP0SkHeEjfe99NehW1tp8nrV1N",
              "marida_w_index": "1qQsnXnSP0SkHeEjfe99NehW1tp8nrV1N"}

def upload_drive(file_path, dataset):
    # Credenciales y configuración
    SCOPES = ['https://www.googleapis.com/auth/drive']
    SERVICE_ACCOUNT_FILE = os.environ['CREDENTIALS']  # Ruta al archivo JSON de credenciales de servicio
    FOLDER_ID = dataset_id[dataset]  # ID de la carpeta destino en Google Drive
    FILE_PATH = file_path  # Ruta al archivo local

    # Nombre del archivo a subir
    file_name = file_path.split('/')[-1]

    # Autenticación usando el archivo de credenciales de servicio
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)

    try:
        # Construir el servicio de Google Drive
        drive_service = build('drive', 'v3', credentials=credentials)

        # Metadata del archivo que se va a subir
        file_metadata = {
            'name': file_name,  # Nombre del archivo en Drive
            'addParents': FOLDER_ID  # ID de la carpeta en la que se subirá el archivo
        }

        # Crear una carga de medios para subir el archivo
        media = MediaFileUpload(FILE_PATH,
                                mimetype='application/csv',  # Tipo MIME del archivo CSV
                                resumable=True)

        # Buscar el archivo existente por nombre en la carpeta destino
        existing_file = drive_service.files().list(q=f"name='{file_name}' and '{FOLDER_ID}' in parents",
                                                   fields='files(id, parents)').execute()

        # Si existe un archivo con el mismo nombre, actualizarlo
        if existing_file.get('files'):
            file_id = existing_file['files'][0]['id']

            # Actualizar los padres del archivo para agregar la carpeta destino
            updated_file = drive_service.files().update(fileId=file_id,
                                                        addParents=FOLDER_ID,
                                                        body=file_metadata,
                                                        media_body=media,
                                                        supportsAllDrives=True).execute()
            print(f"Archivo '{file_name}' actualizado. ID: {updated_file['id']}")
        else:
            # Si no existe, crear un nuevo archivo y añadirlo a la carpeta destino
            file_metadata['addParents'] = FOLDER_ID  # Agregar la carpeta destino al metadata
            new_file = drive_service.files().create(body=file_metadata,
                                                    media_body=media,
                                                    supportsAllDrives=True).execute()
            print(f"Archivo '{file_name}' subido. ID: {new_file['id']}")

    except HttpError as error:
        print(f'Error de HTTP durante la operación en Google Drive: {error}')
    except Exception as e:
        print(f'Ocurrió un error: {e}')


def download_drive(path_dir, subset, dataset):
    # Credenciales y configuración
    SCOPES = ['https://www.googleapis.com/auth/drive']
    SERVICE_ACCOUNT_FILE = os.environ['CREDENTIALS']  # Ruta al archivo JSON de credenciales de servicio
    FOLDER_ID = dataset_id[dataset]  # ID de la carpeta en Google Drive

    # Autenticación usando el archivo de credenciales de servicio
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    # destination_folder, file_name = '/'.join(file_path.split('/')[:-1]), file_path.split('/')[-1]
    destination_folder = f'{path_dir}/reports/'
    file_name = subset+'.csv'
    try:
        # Construir el servicio de Google Drive
        drive_service = build('drive', 'v3', credentials=credentials)

        # Buscar el archivo por nombre en la carpeta destino
        query = f"name='{file_name}' and '{FOLDER_ID}' in parents and trashed=false"
        response = drive_service.files().list(q=query, fields='files(id, name)').execute()
        files = response.get('files', [])

        if not files:
            print(f"No se encontró el archivo: {file_name}")
            return

        file_id = files[0]['id']

        # Descargar el archivo
        request = drive_service.files().get_media(fileId=file_id)
        destination_path = os.path.join(destination_folder, file_name)
        print(destination_path)
        fh = io.FileIO(destination_path, 'wb')
        downloader = MediaIoBaseDownload(fh, request)

        done = False
        while not done:
            status, done = downloader.next_chunk()
            print(f"Descarga {int(status.progress() * 100)}% completa.")

        print(f"Archivo '{file_name}' descargado a '{destination_path}'.")

    except HttpError as error:
        print(f'Error de HTTP durante la operación en Google Drive: {error}')