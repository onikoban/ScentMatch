import gspread
from google.oauth2.service_account import Credentials

def log_to_sheet(data, sheet_id, creds_dict):

    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]

    creds = Credentials.from_service_account_info(
        creds_dict,
        scopes=scope
    )

    client = gspread.authorize(creds)
    sheet = client.open_by_key(sheet_id).sheet1

    sheet.append_row(list(data.values()))
