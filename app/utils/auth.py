import os
import functools
from flask import session, redirect, url_for, request, flash
from app.utils.logging import get_logger

logger = get_logger()

def get_site_password():
    """Get the site password from environment variable"""
    return os.environ.get('AI_WETTER_PASSWORD', 'defaultpassword123')

def is_authenticated():
    """Check if the user is authenticated"""
    return session.get('authenticated', False)

def login_required(f):
    """Decorator to require login for routes"""
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if not is_authenticated():
            flash('Please log in to access this page', 'warning')
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function
