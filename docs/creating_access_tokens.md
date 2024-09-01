# DropBox

Dropbox access tokens are needed to download and upload model weights and datasets straight from this repository.

## DropBox Initialization

1. Set up your DropBox account
    - Go to [dropbox.umich.edu](dropbox.umich.edu).
    - Set up your umich account.
2. Request access to the UMARV DropBox from a team lead.
3. Create your DrobBox developer app.
    - Go to [https://www.dropbox.com/developers/apps](https://www.dropbox.com/developers/apps).
    - Click "Create App".
    - Click "Scoped Access".
    - Click "Full DropBox".
    - Insert your name in the below step.
    - Name the app "UMARV CV {insert_your_name}".
    - Click "I Agree".
    - Click "Create App".
    - Click on the "Permissions" tab.
    - Place checkmarks on the following sections:
        - account_info.read
        - files.metadata.write
        - files.metadata.read
        - files.content.write
        - files.content.read

## DropBox Generate Access Token

1. Go to [https://www.dropbox.com/developers/apps](https://www.dropbox.com/developers/apps)
2. Click on your app.
3. Click "Generate access token".

# GitHub

GitHub access tokens are needed to push changes in the Google Colab and LambdaLabs environments.

## GitHub Generate Access Token

1. Go to [https://github.com/settings/tokens](https://github.com/settings/tokens).
2. Click "Tokens (classic)".
3. Click "Generate new token".
4. Click "Generate new token (classic)".
5. Note : "UMARV CV".
6. Check on "repo".
7. Click "Generate token".
