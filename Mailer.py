import yagmail
from email.message import EmailMessage

class Mailer:
    def __init__(self, email_address, app_password, sender, receivers):
        self.sender = sender
        self.receivers = receivers
        self.email_address = email_address
        self.app_password = app_password

    def send(self, subject = "", message = "", attachment = ""):
        yag = yagmail.SMTP(self.email_address, self.app_password)
        if attachment == "":
            yag.send(self.receivers, subject, contents=message)
        else:
            yag.send(self.receivers, subject, contents=message, attachments=attachment)