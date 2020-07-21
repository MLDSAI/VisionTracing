from flask_script import Manager, prompt_bool, Shell, Server
from termcolor import colored

from app import app


manager = Manager(app)


def make_shell_context():
    return dict(app=app)


manager.add_command('runserver', Server(port=8081))
manager.add_command('shell', Shell(make_context=make_shell_context))


if __name__ == '__main__':
    manager.run()
