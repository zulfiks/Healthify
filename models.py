from extensions import db

# 1. Tabel Laporan (UPDATE: Ada Image)
class Laporan(db.Model):
    __tablename__ = 'laporan_kendala' 
    id = db.Column(db.Integer, primary_key=True)
    pengguna = db.Column(db.String(100), nullable=False)
    tanggal = db.Column(db.String(20), nullable=False)
    deskripsi = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(20), default='Diproses')
    image = db.Column(db.String(255), nullable=True) # <--- BARU: Screenshot
    
    def to_dict(self):
        return { 
            'id': self.id, 
            'pengguna': self.pengguna, 
            'tanggal': self.tanggal, 
            'deskripsi': self.deskripsi, 
            'status': self.status,
            'image': self.image 
        }

# 2. Tabel Admin (Tetap)
class Admin(db.Model):
    __tablename__ = 'admin'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    nama = db.Column(db.String(100), default='Admin')
    def to_dict(self):
        return { 'id': self.id, 'email': self.email, 'nama': self.nama }

# 3. Tabel Konten (Tetap)
class Konten(db.Model):
    __tablename__ = 'konten_edukasi'
    id = db.Column(db.Integer, primary_key=True)
    judul = db.Column(db.String(200), nullable=False)
    kategori = db.Column(db.String(100), nullable=False)
    publikasi = db.Column(db.String(20), nullable=False)
    tautan = db.Column(db.String(255), nullable=False)
    def to_dict(self):
        return { 'id': self.id, 'judul': self.judul, 'kategori': self.kategori, 'publikasi': self.publikasi, 'tautan': self.tautan }

# 4. Tabel Pengguna (UPDATE: Ada Poin)
class Pengguna(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    nama = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    umur = db.Column(db.Integer)
    gender = db.Column(db.String(20))
    tinggi = db.Column(db.Integer)
    berat = db.Column(db.Integer)
    poin = db.Column(db.Integer, default=0) # <--- BARU: Untuk Leaderboard
    
    def to_dict(self):
        return { 
            'id': self.id, 
            'nama': self.nama, 
            'email': self.email, 
            'umur': self.umur, 
            'gender': self.gender, 
            'tinggi': self.tinggi, 
            'berat': self.berat,
            'poin': self.poin
        }