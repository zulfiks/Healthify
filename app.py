from flask import Flask, render_template, redirect, url_for, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import os
import json
from flask_cors import CORS 

# --- KONFIGURASI DB & APLIKASI ---

# Ganti 'password_anda' dan 'healthify' sesuai setup MySQL Anda
DB_URI = 'mysql+pymysql://root:@127.0.0.1:3306/healthify'

app = Flask(__name__)
app.config['SECRET_KEY'] = 'kunci_rahasia_anda_yang_kuat_dan_unik' 
app.config['SQLALCHEMY_DATABASE_URI'] = DB_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

CORS(app) 
db = SQLAlchemy(app)

# --- MODEL DATA (PENGGANTI ARRAY/TABEL ADMIN) ---

class Admin(db.Model):
    """Merepresentasikan tabel 'admin' di MySQL."""
    __tablename__ = 'admin' 
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), nullable=False, unique=True)
    password_hash = db.Column(db.String(255), nullable=False)
    
    # Fungsi pembantu untuk mengkonversi Model ke Dictionary (untuk respons JSON)
    def to_dict(self):
        return {
            'id': self.id,
            'email': self.email,
            # JANGAN SERTAKAN password_hash dalam respons API publik!
        }
        
# Membuat tabel di database (Hanya perlu dijalankan sekali)
with app.app_context():
    db.create_all()

# --- HELPER LOAD DATA FILE ---

def load_csv_data(filename):
    """Memuat data dari file CSV menggunakan Pandas."""
    data_path = os.path.join(app.root_path, 'data_files', filename)
    try:
        df = pd.read_csv(data_path)
        return df.to_dict(orient="records")
    except Exception as e:
        print(f"Error memuat CSV: {e}")
        return []

def load_json_data(filename):
    """Memuat data dari file JSON."""
    data_path = os.path.join(app.root_path, 'data_files', filename)
    try:
        with open(data_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error memuat JSON: {e}")
        return []

# --- HELPER OTENTIKASI (MENGGUNAKAN MODEL BARU) ---

def check_login_from_db(email, password):
    """Mengecek kredensial admin menggunakan SQLAlchemy."""
    admin = Admin.query.filter_by(
        email=email.strip(), 
        password_hash=password.strip()
    ).first()
    
    return admin # Mengembalikan objek Admin jika ditemukan, None jika tidak.

# --- RUTE CRUD API (UNTUK MENGELOLA ADMIN) ---

# READ ONE (GET)
@app.route('/api/admin/<int:user_id>', methods=['GET'])
def get_admin_detail(user_id):
    """Mengambil detail satu admin."""
    admin = Admin.query.get_or_404(user_id)
    return jsonify(admin.to_dict())

# UPDATE (PUT)
@app.route('/api/admin/<int:user_id>', methods=['PUT'])
def update_admin(user_id):
    """Mengedit data admin yang sudah ada."""
    admin = Admin.query.get_or_404(user_id)
    # Memerlukan data JSON dari request body Postman
    data = request.get_json() 
    
    # Memperbarui hanya data yang tersedia di body request
    if 'nama' in data:
        admin.nama = data['nama']
    if 'email' in data:
        admin.email = data['email']
    
    db.session.commit() # Menyimpan perubahan ke database
    return jsonify({"message": "Admin updated successfully", "user": admin.to_dict()}), 200

# DELETE (DELETE)
@app.route('/api/admin/<int:user_id>', methods=['DELETE'])
def delete_admin(user_id):
    """Menghapus data admin."""
    admin = Admin.query.get_or_404(user_id)
    
    db.session.delete(admin)
    db.session.commit()
    return jsonify({"message": f"Admin ID {user_id} deleted successfully"}), 200

# --- RUTE OTENTIKASI & TAMPILAN ---

@app.route('/authenticate', methods=['POST'])
def authenticate():
    """Rute API untuk proses login, merespons dengan JSON."""
    email = request.form.get('email')
    password = request.form.get('password')
    
    if not email or not password:
        return jsonify({'status': 'error', 'message': 'Email dan password harus diisi.'}), 400
        
    user_data = check_login_from_db(email, password)
    
    if user_data:
        # Login Berhasil
        return jsonify({
            'status': 'success',
            'message': 'Login berhasil',
            'user_id': user_data.id,
            'redirect_to': '/kelola-makanan'
        }), 200
    else:
        # Login Gagal
        return jsonify({'status': 'error', 'message': 'Email atau password salah.'}), 401

@app.route('/')
@app.route('/login', methods=['GET'])
def login():
    """Menampilkan halaman login."""
    return render_template('login.html', title='Login Admin')

@app.route("/kelola-makanan")
def kelolamakanan():
    """Menampilkan data CSV Makanan (HTML)."""
    makanan_data = load_csv_data('data.csv')
    return render_template("kelolamakanan.html", data=makanan_data, title="Kelola Data Makanan")

@app.route("/laporan-kendala")
def laporankendala():
    """Menampilkan data JSON Laporan Kendala (HTML)."""
    data_kendala = load_json_data('laporan_kendala.json')
    return render_template("laporan-kendala.html", laporan=data_kendala, title="Laporan Kendala")
    
@app.route('/api/laporan', methods=['GET'])
def get_laporan():
    """Mengirim data kendala sebagai JSON mentah."""
    data_kendala = load_json_data('laporan_kendala.json')
    return jsonify(data_kendala)

if __name__ == "__main__":
    # Jalankan server. Pastikan MySQL Anda sudah aktif!
    app.run(debug=True, port=5000)