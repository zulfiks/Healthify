from flask import Flask, jsonify, request
from flask_cors import CORS 
from extensions import db
from models import Laporan, Admin, Konten, Pengguna 
import pandas as pd
import os

app = Flask(__name__)

# --- KONFIGURASI ---
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/healthify'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'kunci_rahasia'

CORS(app, resources={r"/api/*": {"origins": "*"}}) 

db.init_app(app)

# --- BUAT TABEL ---
with app.app_context():
    db.create_all()

# ==========================================
# API LOGIN
# ==========================================
@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    
    if not email or not password:
        return jsonify({"message": "Wajib diisi"}), 400
    
    admin = Admin.query.filter_by(email=email).first()
    
    if admin and admin.password_hash == password:
        return jsonify({
            "status": "success", 
            "message": "Login Berhasil!", 
            "user": admin.to_dict()
        }), 200
    else:
        return jsonify({"message": "Email/Password salah"}), 401

# ==========================================
# API PROFIL ADMIN (BARU)
# ==========================================
@app.route('/api/admin/profile', methods=['GET'])
def get_profile():
    email = request.args.get('email')
    if not email:
        return jsonify({"message": "Email required"}), 400
        
    admin = Admin.query.filter_by(email=email).first()
    if admin:
        return jsonify(admin.to_dict())
    return jsonify({"message": "Admin not found"}), 404

@app.route('/api/admin/profile', methods=['PUT'])
def update_profile():
    data = request.get_json()
    email_lama = data.get('email_lama')
    
    admin = Admin.query.filter_by(email=email_lama).first()
    if not admin:
        return jsonify({"message": "Admin not found"}), 404
        
    if 'nama' in data: admin.nama = data['nama']
    if 'password' in data and data['password']: admin.password_hash = data['password']
        
    db.session.commit()
    return jsonify({"message": "Updated", "user": admin.to_dict()}), 200

# ==========================================
# API USERS (Manajemen Akun)
# ==========================================
@app.route('/api/users', methods=['GET'])
def get_users():
    return jsonify([u.to_dict() for u in Pengguna.query.all()]), 200

@app.route('/api/users', methods=['POST'])
def add_user():
    data = request.get_json()
    new = Pengguna(nama=data['nama'], email=data['email'], umur=data['umur'], gender=data['gender'], tinggi=data['tinggi'], berat=data['berat'])
    db.session.add(new)
    db.session.commit()
    return jsonify({"message": "Added", "data": new.to_dict()}), 201

@app.route('/api/users/<int:id>', methods=['PUT'])
def update_user(id):
    user = Pengguna.query.get_or_404(id)
    data = request.get_json()
    user.nama = data['nama']
    user.email = data['email']
    user.umur = data['umur']
    user.gender = data['gender']
    user.tinggi = data['tinggi']
    user.berat = data['berat']
    db.session.commit()
    return jsonify({"message": "Updated", "data": user.to_dict()}), 200

@app.route('/api/users/<int:id>', methods=['DELETE'])
def delete_user(id):
    user = Pengguna.query.get_or_404(id)
    db.session.delete(user)
    db.session.commit()
    return jsonify({"message": "Deleted"}), 200

# ==========================================
# API MAKANAN (CSV)
# ==========================================
def get_csv_path():
    return os.path.join(app.root_path, 'data_files', 'data.csv')

@app.route('/api/makanan', methods=['GET'])
def get_makanan():
    try:
        df = pd.read_csv(get_csv_path()).fillna('')
        return jsonify(df.to_dict(orient='records')), 200
    except: return jsonify([]), 200

@app.route('/api/makanan', methods=['POST'])
def add_makanan():
    try:
        csv = get_csv_path()
        df = pd.read_csv(csv)
        data = request.get_json()
        new_id = df['id'].max() + 1 if not df.empty else 1
        new_row = {'id': new_id, 'name': data['name'], 'calories': data['calories'], 'proteins': data['proteins'], 'fat': data['fat'], 'carbohydrate': data['carbohydrate'], 'image': data['image']}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(csv, index=False)
        return jsonify({"message": "Added"}), 201
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route('/api/makanan/<int:id>', methods=['PUT'])
def update_makanan(id):
    try:
        csv = get_csv_path()
        df = pd.read_csv(csv)
        data = request.get_json()
        if id not in df['id'].values: return jsonify({"message": "404"}), 404
        idx = df.index[df['id'] == id].tolist()[0]
        df.at[idx, 'name'] = data['name']
        df.at[idx, 'calories'] = data['calories']
        df.at[idx, 'proteins'] = data['proteins']
        df.at[idx, 'fat'] = data['fat']
        df.at[idx, 'carbohydrate'] = data['carbohydrate']
        df.at[idx, 'image'] = data['image']
        df.to_csv(csv, index=False)
        return jsonify({"message": "Updated"}), 200
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route('/api/makanan/<int:id>', methods=['DELETE'])
def delete_makanan(id):
    try:
        csv = get_csv_path()
        df = pd.read_csv(csv)
        df = df[df['id'] != id]
        df.to_csv(csv, index=False)
        return jsonify({"message": "Deleted"}), 200
    except Exception as e: return jsonify({"error": str(e)}), 500

# ==========================================
# API KONTEN & LAPORAN
# ==========================================
@app.route('/api/konten', methods=['GET'])
def get_konten():
    return jsonify([i.to_dict() for i in Konten.query.all()]), 200

@app.route('/api/konten', methods=['POST'])
def add_konten():
    d = request.get_json()
    new = Konten(judul=d['judul'], kategori=d['kategori'], publikasi=d['publikasi'], tautan=d['tautan'])
    db.session.add(new)
    db.session.commit()
    return jsonify({"message": "Added"}), 201

@app.route('/api/konten/<int:id>', methods=['PUT'])
def update_konten(id):
    item = Konten.query.get_or_404(id)
    d = request.get_json()
    item.judul = d['judul']; item.kategori = d['kategori']; item.publikasi = d['publikasi']; item.tautan = d['tautan']
    db.session.commit()
    return jsonify({"message": "Updated"}), 200

@app.route('/api/konten/<int:id>', methods=['DELETE'])
def delete_konten(id):
    db.session.delete(Konten.query.get_or_404(id))
    db.session.commit()
    return jsonify({"message": "Deleted"}), 200

@app.route('/api/laporan', methods=['GET'])
def get_laporan():
    return jsonify([d.to_dict() for d in Laporan.query.all()])

if __name__ == "__main__":
    app.run(debug=True, port=5000)