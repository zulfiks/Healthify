from flask_sqlalchemy import SQLAlchemy

# Kita bikin objek db di sini biar tidak error "circular import"
db = SQLAlchemy()