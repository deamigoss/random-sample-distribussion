import altair as alt
import numpy as np
import pandas as pd
from scipy.stats import norm, binom, expon, uniform
import streamlit as st

X_INTERVAL = 1000

def generate_altair_pdf(df):
    """Membuat dan mengembalikan bagan garis Altair pdf (probability density function) dari Pandas DataFrame
     berisi beberapa array x dan array nilai yang sesuai dengan nilai pdf yang diberikan, f(x), pada setiap x"""
    pdf_line = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x='x',
            y='f(x)'
        )
    )

    nearest = alt.selection(
        type='single', 
        nearest=True, 
        on='mouseover', 
        fields=['x'], 
        empty='none'
    )

    selectors = (
        alt.Chart(df)
        .mark_point()
        .encode(
            x='x:Q',
            opacity=alt.value(0),
        )
        .add_selection(nearest)
    )

    points = (
        pdf_line
        .mark_point()
        .encode(
            opacity=(
                alt.condition(
                    nearest, 
                    alt.value(1), 
                    alt.value(0)
                )
            )
        )
    )

    text = (
        pdf_line
        .mark_text(
            align='left', 
            dx=5, 
            dy=-5, 
            color='white'
        )
        .encode(
            text=(
                alt.condition(
                    nearest, 
                    'f(x):Q', 
                    alt.value(' ')
                )
            )
        )
    )

    rules = (
        alt.Chart(df)
        .mark_rule(color='gray')
        .encode(x='x:Q',)
        .transform_filter(nearest)
    )

    pdf_chart = alt.layer(
        pdf_line, 
        selectors, 
        points, 
        rules, 
        text
    )

    return pdf_chart


def generate_altair_pmf(df):
    """Membuat dan mengembalikan diagram titik Altair (probability mass function) pmf dari Pandas DataFrame
     berisi beberapa array x dan array nilai yang sesuai dengan nilai pmf yang diberikan, p(x), pada setiap x"""

    base = alt.Chart(df)
    
    points = (
        base
        .mark_point()
        .encode(
            x='x',
            y='p(x)',
            size=alt.value(90),
            fill=alt.value('#4682b4'),
            tooltip=['p(x)']
        )
    )

    vlines = (
        base
        .mark_rule()
        .encode(
            x='x',
            y='p(x)',
            color=alt.value('white'),
            strokeWidth=alt.value(2)
        )
    )

    pmf_chart = alt.layer(
        points,
        vlines
    )

    return pmf_chart


def generate_altair_sample_hist(sample):
    """Membuat dan mengembalikan bagan histogram Altair dari Pandas DataFrame"""
    base = alt.Chart(sample)

    max_bins = st.slider('Max histogram bins', 5, 40, 40)

    sample_hist = (
        base
        .mark_bar()
        .encode(
            x=alt.X(
                'vals:Q', 
                bin={"maxbins": max_bins}
            ),
            y='count()'
        )
        .properties(
            title=f'Random Sample (n = {len(sample)})'
        )
    )

    sample_hist = (
        sample_hist
        .configure_title(
            fontSize=20,
            font='Courier New'
        )
    )

    return sample_hist


def main():
    # Pilihan Distribusi
    st.sidebar.subheader("Distribusi")
    distribution = st.sidebar.selectbox(
        label='Pilih distribusi', 
        options=('Normal', 'Binomial', 'Exponensial', 'Uniform')
    )

    # Pilh Ukuran Sampel
    sample_size = st.sidebar.slider(
        label='Pilih ukuran sampel', 
        min_value=1, 
        max_value=1000, 
        value=100
    )
 
    st.sidebar.subheader('Parameter')
    if distribution == "Normal":
        # Normal Parameter
        input_type = st.sidebar.radio(
            'Input Type',
            ('Pilih Nilai', 'Masukkan Nilai Manual')
        )
        if input_type == 'Pilih Nilai':
            mu = st.sidebar.slider(
                label='\u03BC (Mean)', 
                min_value=-10.0, 
                max_value=10.0, 
                value=0.0
            )
            sigma_squared = st.sidebar.slider(
                label='\u03C3\u00B2 (Variance)', 
                min_value=0.01, 
                max_value=10.0, 
                value=1.0
            )
        elif input_type == 'Masukkan Nilai Manual':
            mu = st.sidebar.number_input(
                label='\u03BC (Mean)', 
                value=0.0, 
                step=1.0
            )
            sigma_squared = st.sidebar.number_input(
                label='\u03C3\u00B2 (Variance)', 
                min_value=0.01, 
                value=1.0, 
                step=0.1
            )

        # Normal PDF
        x = np.linspace(
            norm.ppf(0.001, mu, sigma_squared), 
            norm.ppf(0.999, mu, sigma_squared),
            X_INTERVAL
        )
        df = pd.DataFrame({
            'x': x, 
            'f(x)': norm.pdf(x, mu, sigma_squared)
        })
        # Normal PDF line chart
        norm_pdf_chart = generate_altair_pdf(df)
        # Tampilan di Web
        st.latex('PDF\\ of\\ \mathcal{N}'+f'({np.round(mu, 2)}, {np.round(sigma_squared, 2)})')
        st.altair_chart(
            norm_pdf_chart, 
            use_container_width=True
        )
        
        # Sampel Random
        sample = pd.DataFrame(
            norm.rvs(loc=mu, scale=sigma_squared, size=sample_size), 
            columns=['vals']
        )
        # Sampel Histogram
        sample_hist = generate_altair_sample_hist(sample)
        # Tampilan di Web
        st.altair_chart(
            sample_hist, 
            use_container_width=True
        )

   
    elif distribution == "Binomial":
        # Binomial Parameter
        input_type = st.sidebar.radio(
            'Input Type',
            ('Pilih Nilai', 'Masukkan Nilai Manual')
        )
        if input_type == 'Pilih Nilai':
            n = st.sidebar.slider(
                label='n', 
                min_value=1, 
                max_value=100, 
                value=20
            )
            p = st.sidebar.slider(
                label='p', 
                min_value=0.00, 
                max_value=1.00, 
                value=0.50
            )
        elif input_type == 'Masukkan Nilai Manual':
            n = st.sidebar.number_input(
                label='n', 
                min_value=1,  
                value=20,
                step=1
            )
            p = st.sidebar.number_input(
                label='p', 
                min_value=0.00,
                max_value=1.0,
                value=0.50,
                step=0.01
            )

        # Binomial PMF
        x = np.arange(
            binom.ppf(0.001, n, p), 
            binom.ppf(0.999, n, p) + 1
        )
        df = pd.DataFrame({
                'x': x, 
                'p(x)': binom.pmf(x, n, p)
        })
        
        # Binomial PMF chart
        binomial_pmf_chart = generate_altair_pmf(df)
        # Tampilan di Web
        st.latex('PMF\\ of\\ \mathcal{Binom}'+f'({n}, {np.round(p, 2)})')
        st.altair_chart(
            binomial_pmf_chart, 
            use_container_width=True
        )

        # Sampel Random
        sample = pd.DataFrame(
            binom.rvs(n, p, size=sample_size), 
            columns=['vals']
        )
        # Sampel Histogram
        sample_hist = generate_altair_sample_hist(sample)
        # Tampilan di Web
        st.altair_chart(
            sample_hist, 
            use_container_width=True
        )

    elif distribution == "Exponensial":
        # Exponensial Parameter
        input_type = st.sidebar.radio(
            'Input Type',
            ('Pilih Nilai', 'Masukkan Nilai Manual')
        )
        if input_type == 'Pilih Nilai':
            l = st.sidebar.slider(
                label='\u03BB', 
                min_value=0.01, 
                max_value=10.0, 
                value=1.0
            )
        elif input_type == 'Masukkan Nilai Manual':
            l = st.sidebar.number_input(
                label='\u03BB', 
                min_value=0.01, 
                value=1.0,
                step=0.1
            )

        # Exponenensial PDF
        x = np.linspace(
            expon.ppf(0.001, scale=1/l), 
            expon.ppf(0.999, scale=1/l), 
            X_INTERVAL
        )
        df = pd.DataFrame({
            'x': x, 
            'f(x)': expon.pdf(x, scale=1/l)
        })
        # Exponensial PDF line chart
        expon_pdf_chart = generate_altair_pdf(df)
        # Tampilan di Web
        st.latex('PDF\\ of\\ Exp'+f'({np.round(l, 2)})')
        st.altair_chart(
            expon_pdf_chart, 
            use_container_width=True
        )

        # Sampel Random
        sample = pd.DataFrame(
            expon.rvs(scale=1/l, size=sample_size), 
            columns=['vals']
        )
        # Sampel Histogram
        sample_hist = generate_altair_sample_hist(sample)
        # Tampilan di Web
        st.altair_chart(
            sample_hist, 
            use_container_width=True
        )

    elif distribution == "Uniform":
        # Uniform Parameter
        a = 0
        b = 1
        input_type = st.sidebar.radio(
            'Input Type',
            ('Pilih Nilai', 'Masukkan Nilai Manual')
        )
        if input_type == 'Pilih Nilai':
            a = st.sidebar.slider(
                label='a', 
                min_value=0, 
                max_value=9, 
                value=0
            )
            b = st.sidebar.slider(
                label='b', 
                min_value=1, 
                max_value=10, 
                value=1
            )
        elif input_type == 'Masukkan Nilai Manual':
            a = st.sidebar.number_input(
                label='a',
                value=0,
                step=1
            )
            b = st.sidebar.number_input(
                label='b', 
                value=1,
                step=1
            ) 
        
        try:
            # Uniform PDF
            x = np.linspace(
                uniform.ppf(0.001, loc=a, scale=b-a), 
                uniform.ppf(0.999, loc=a, scale=b-a), 
                X_INTERVAL
            )
            df = pd.DataFrame({
                'x': x, 
                'f(x)': uniform.pdf(x, loc=a, scale=b-a)
            })
            # Uniform PDF line chart
            uniform_pdf_chart = generate_altair_pdf(df)
            # Tampilan di Web
            st.latex(f'PDF\\ of\\ U([{a}, {b}]))')
            st.altair_chart(
                uniform_pdf_chart, 
                use_container_width=True
            )

            # Sampel Random
            sample = pd.DataFrame(
                uniform.rvs(loc=a, scale=b-a, size=sample_size), 
                columns=['vals']
            )
            # Sampel Histogram
            sample_hist = generate_altair_sample_hist(sample)
            # Tampilan di Web
            st.altair_chart(
                sample_hist, 
                use_container_width=True
            )
        except ValueError:
            # Notifaksi pengguna jika salah memasukkan parameter
            st.warning('Maaf tidak bisa menampilkan grafik - Parameter "a" harus lebih kecil atau sama dengan "b"')

st.sidebar.title('Sampel random')
st.sidebar.write('1. Pilih distrubsi')
st.sidebar.write('2. Pilih ukuran sampel dan parameter')
main()
