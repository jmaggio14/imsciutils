# @Email: jmaggio14@gmail.com
# @Website: https://www.imagepypelines.org/
# @License: https://github.com/jmaggio14/imsciutils/blob/master/LICENSE
# @github: https://github.com/jmaggio14/imsciutils
#
# Copyright (c) 2018 Jeff Maggio, Nathan Dileas, Ryan Hartzell
import smtplib
import getpass
from .. import core

from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart


class Emailer(object):
    """
    Goal is to build an object which can be used to automatically send emails
    after a test or run completes.
    simplicity trumps functionality here, we mostly want it to be easy to use

    Example:
        >>> import os
        >>> with Emailer(os.environ['GMAIL_USER'], [os.environ['GMAIL_USER']]) as emailer:
        ...     emailer.body('this is a test:\n\n\n\nblah -nate')
        ...     emailer.send(os.environ['GMAIL_PASS'])
        >>>
    """

    def __init__(self,
                sender,
                recipients,
                subject="noreply: imsciutils automated email",
                server_name='smtp.gmail.com',
                server_port=465):
        self.subject = subject

        # TODO verify that recipients are valid here
        # ND: what is the rationale here?
        # JM: for a line in get_msg: self.current_msg['To'] = ', '.join(self.recipients)
        # my thinking a list or a single address can be passed in, it's admittedly a lil awk
        if isinstance(recipients, str):
            recipients = [recipients]

        self.sender = sender
        self.recipients = recipients
        self.subject = subject
        self.current_msg = None

        self.server_name = server_name
        self.server_port = server_port

    def get_msg(self):
        """
        returns the current email message or creates a new one if one
        is not already queued
        """
        if self.current_msg is not None:
            return self.current_msg

        self.current_msg = MIMEMultipart('alternative')
        self.current_msg['Subject'] = self.subject
        self.current_msg['To'] = ', '.join(self.recipients)
        self.current_msg['From'] = self.sender
        return self.current_msg

    def attach(self, filename):
        """
        attaches a file to the email message
        """
        msg = self.get_msg()

        if not os.path.isfile(filename):
            core.error("file '{}' does not exist or is inaccessible,\
                            skipping attachment!".format(filename))
            return

        with open(filename, 'rb') as fp:
            msg.attach(fp.read())

    def body(self, text):
        """
        sets the body of the current email message
        """
        if not isinstance(text, str):
            ie.error("unable to set body because text must be a str,\
                    currently".format(type(text)))
            return

        msg = self.get_msg()
        msg.attach(MIMEText(text, 'plain'))

    def send(self, password=None):
        """
        sends the current message and clears the template so a new
        message can be created
        """
        if password is None:
            password = getpass.getpass()

        msg = self.get_msg()

        server = smtplib.SMTP_SSL(self.server_name, self.server_port)
        server.ehlo()
        server.login(self.sender, password)
        server.send_message(msg)

        self.current_msg = None

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()


def main():
    import os
    with Emailer(os.environ['GMAIL_USER'], [os.environ['GMAIL_USER']]) as emailer:
        emailer.body('this is a test:\n\n\n\nblah -nate')
        emailer.send(os.environ['GMAIL_PASS'])


if __name__ == "__main__":
    main()
