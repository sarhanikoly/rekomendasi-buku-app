from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)
app.secret_key = 'projekRekomenBuku'

# ======= Load Data Buku =======
with open('books.json', 'r', encoding='utf-8') as f:
    books = json.load(f)

df = pd.DataFrame(books)
df['judul'] = df['judul'].fillna('')

# ======= TF-IDF + Cosine Similarity =======
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['judul'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# ======= Load Data User =======
if os.path.exists('users.json'):
    with open('users.json', 'r', encoding='utf-8') as f:
        users = json.load(f)
else:
    users = {'admin': '12345', 'desvita': '122106'}

# ======= Load Data Favorit =======
if os.path.exists('favorit.json'):
    with open('favorit.json', 'r', encoding='utf-8') as f:
        favorit = json.load(f)
else:
    favorit = {}

# ======= Halaman Utama =======
@app.route('/')
def home():
    if 'username' not in session:
        return redirect(url_for('login'))

    genres = sorted(set(df['genre']))
    return render_template('home.html', username=session['username'], genres=genres)

# ======= Login =======
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = request.form['username']
        pw = request.form['password']
        if user in users and users[user] == pw:
            session['username'] = user
            return redirect(url_for('home'))
        return render_template('login.html', error='Username atau password salah')
    return render_template('login.html')

# ======= Register =======
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm = request.form['confirm_password']

        if username in users:
            return render_template('register.html', error='Username sudah digunakan.')
        if password != confirm:
            return render_template('register.html', error='Password tidak cocok.')

        users[username] = password
        with open('users.json', 'w', encoding='utf-8') as f:
            json.dump(users, f)

        session['username'] = username
        return redirect(url_for('home'))

    return render_template('register.html')

# ======= Logout =======
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

# ======= Detail Buku =======
@app.route('/detail/<int:id>')
def detail(id):
    if 'username' not in session:
        return redirect(url_for('login'))

    buku = df[df['id'] == id].squeeze()
    username = session['username']
    user_favorit = favorit.get(username, [])

    return render_template('detail.html', buku=buku, favorites=favorit, username=username)

# ======= Rekomendasi Buku =======
@app.route('/rekomendasi/<int:id>')
def rekomendasi(id):
    if 'username' not in session:
        return redirect(url_for('login'))

    idx = df[df['id'] == id].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    rekomendasi_idx = [i[0] for i in sim_scores]
    hasil = df.iloc[rekomendasi_idx]

    buku_awal = df.iloc[idx]
    return render_template('rekomendasi.html', buku_awal=buku_awal, rekomendasi=hasil.to_dict(orient='records'))

# ======= Pencarian Buku =======
@app.route('/cari')
def cari():
    if 'username' not in session:
        return redirect(url_for('login'))

    keyword = request.args.get('q', '').lower()
    hasil = df[df['judul'].str.lower().str.contains(keyword)]

    username = session['username']

    if hasil.empty:
        genre_favorit = get_genre_favorit(username)
        rekomendasi_ai = get_ai_rekomendasi(genre_favorit) if genre_favorit else []
    else:
        genre_favorit = None
        rekomendasi_ai = []

    return render_template(
        'pencarian.html',
        books=hasil.to_dict(orient='records'),
        keyword=keyword,
        genre_favorit=genre_favorit,
        rekomendasi_ai=rekomendasi_ai
    )

# ======= Filter Berdasarkan Genre =======
@app.route('/genre/<nama_genre>')
def genre(nama_genre):
    if 'username' not in session:
        return redirect(url_for('login'))

    hasil = df[df['genre'].str.lower() == nama_genre.lower()]
    return render_template('genre.html', books=hasil.to_dict(orient='records'), genre=nama_genre)

# ======= Tambah/Hapus Buku Favorit =======
@app.route('/favorit/<int:id>', methods=['POST'])
def favorit_toggle(id):
    if 'username' not in session:
        return redirect(url_for('login'))

    username = session['username']
    if username not in favorit:
        favorit[username] = []

    if id in favorit[username]:
        favorit[username].remove(id)
    else:
        favorit[username].append(id)

    with open('favorit.json', 'w', encoding='utf-8') as f:
        json.dump(favorit, f)

    return redirect(url_for('detail', id=id))

# ======= Hapus dari Halaman Favorit =======
@app.route('/hapus_favorit/<int:id>', methods=['POST'])
def hapus_favorit(id):
    if 'username' not in session:
        return redirect(url_for('login'))

    username = session['username']
    if username in favorit and id in favorit[username]:
        favorit[username].remove(id)
        with open('favorit.json', 'w', encoding='utf-8') as f:
            json.dump(favorit, f)

    return redirect(url_for('favoritku'))

# ======= Halaman Daftar Favorit =======
@app.route('/favoritku')
def favoritku():
    if 'username' not in session:
        return redirect(url_for('login'))

    username = session['username']
    ids = favorit.get(username, [])
    fav_books = df[df['id'].isin(ids)].to_dict(orient='records')
    return render_template('favoritku.html', books=fav_books, username=username)

def get_genre_favorit(username):
    user_fav_ids = favorit.get(username, [])
    fav_books = df[df['id'].isin(user_fav_ids)]
    return fav_books['genre'].mode()[0] if not fav_books.empty else None

@app.route('/profil')
def profil():
    if 'username' not in session:
        return redirect(url_for('login'))

    username = session['username']
    fav_ids = favorit.get(username, [])
    fav_books = df[df['id'].isin(fav_ids)]
    genre_fav = get_genre_favorit(username)

    return render_template('profil.html', 
                           username=username,
                           jumlah_favorit=len(fav_ids),
                           genre_favorit=genre_fav,
                           buku_favorit=fav_books.to_dict(orient='records'))


@app.route('/rekomendasi_ai')
def rekomendasi_ai():
    if 'username' not in session:
        return redirect(url_for('login'))

    username = session['username']
    genre_utama = get_genre_favorit(username)

    if genre_utama:
        data_genre = df[df['genre'] == genre_utama]
        rekomendasi = data_genre.sample(n=min(5, len(data_genre)))
    else:
        rekomendasi = pd.DataFrame()  # kosongkan jika tidak ada genre

    return render_template(
        'rekomendasi_ai.html',
        rekomendasi=rekomendasi.to_dict(orient='records'),
        genre=genre_utama
    )
def get_ai_rekomendasi(genre):
    data_genre = df[df['genre'].str.lower() == genre.lower()]
    hasil = data_genre.sample(n=min(5, len(data_genre)))
    return hasil.to_dict(orient='records')  # <-- ini yang penting

# ======= Jalankan Aplikasi =======
if __name__ == '__main__':
    app.run(debug=True)
