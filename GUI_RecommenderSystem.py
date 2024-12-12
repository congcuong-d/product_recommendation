# !pip install surprise

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datetime import datetime
import joblib
from surprise import KNNBasic, Reader, Dataset
# from PIL import Image


# Cấu hình trang
st.sidebar.image("hasaki_banner_5.jpg", use_container_width=True)
# st.title("Project 2: Recommender System")
st.title("HASAKI RECOMMENDER SYSTEM")

# Using menu
menu = ["Business Objective","Build Project","Recommend New Product"]
choice = st.sidebar.selectbox('Menu', menu) # Gán menu vào sidebar
st.sidebar.write("""#### 🟢 Thành viên thực hiện: 
                 Hoàng Ngọc Thuỷ Thương
    Lê Duy Quang 
    Đinh Công Cường""")
st.sidebar.write("""#### 🟡 Giáo viên hướng dẫn: 
                 Cô Khuất Thuỳ Phương""")
current_time = datetime.now()
st.sidebar.write(f"""#### 🗓️ Thời gian báo cáo:
                 {current_time.strftime('%d-%m-%Y')}""")


if choice == 'Business Objective':    
    # Thực hiện các công việc khi người dùng chọn Home
    st.subheader("Business Objective") 
    st.markdown(
    """
    Xây dựng hệ thống đề xuất để hỗ trợ người dùng nhanh chóng lựa chọn được sản phẩm phù hợp trên Hasaki
    👈 ***lựa chọn phần đề xuất trên sidebar*** để xem chi tiết.
    ### Hệ thống sẽ gồm hai mô hình gợi ý chính:
    - Collaborative filtering (Surprise)
    - Content-based filtering (Cosine Similarity)   
    """)
    st.image("Content-based-filtering-vs-Collaborative-filtering-Source.jpg", use_container_width=True)



elif choice == 'Build Project':
    # Thuật toán 1. Collaborative filtering (Surprise)
    st.subheader("I. Collaborative filtering (Surprise)")

    # 1. Chuẩn bị dữ liệu
    st.markdown("##### 1. Chuẩn bị dữ liệu")
    st.markdown("""
    - **Đọc dữ liệu**: Tải dữ liệu từ file CSV để xây dựng mô hình.
    - **Xử lý dữ liệu thiếu**: Loại bỏ các dòng dữ liệu có giá trị `null` trong các cột cần thiết như `ma_khach_hang_idx`, `ma_san_pham_idx`, và `so_sao`.
    - **Kiểm tra và lọc dữ liệu**:
      - Kiểm tra giá trị trong cột `so_sao` để đảm bảo dữ liệu đánh giá nằm trong khoảng hợp lệ (1–5).
      - Lọc lại dữ liệu để đảm bảo chỉ giữ các giá trị nằm trong khoảng này.
    - **Sử dụng thư viện Surprise**:
      - Chuyển đổi dữ liệu về dạng thích hợp cho Surprise bằng cách sử dụng `Reader` với `rating_scale=(1, 5)`.
    """)

    # 2. Khám phá dữ liệu
    st.markdown("##### 2. Khám phá dữ liệu")
    st.markdown("""
    - Hiển thị thông tin cơ bản như số lượng người dùng (`n_users`), số lượng sản phẩm (`n_items`) trong tập dữ liệu.
    """)

    # 3. Đánh giá các thuật toán
    st.markdown("##### 3. Đánh giá các thuật toán")
    st.markdown("""
    - **Danh sách các thuật toán đã triển khai**:
      - KNNBaseline
      - SVD (Singular Value Decomposition)
      - SVDpp (SVD++)
      - BaselineOnly
    - **Quy trình đánh giá**:
      - Sử dụng `cross_validate` với các chỉ số RMSE và MAE để đánh giá các thuật toán trên 5 lần gập (5-fold cross-validation).
      - Ghi lại thời gian huấn luyện và kết quả đánh giá (RMSE, MAE).
      - Kết quả được lưu vào DataFrame `results_df` để so sánh các thuật toán.
    """)
    st.image("Bảngđánhgiáthuậttoán.png", caption="Bảng đánh giá thuật toán", use_container_width=True)

    # 4. Tối ưu hóa siêu tham số
    st.markdown("##### 4. Tối ưu hóa siêu tham số")
    st.markdown("""
    - **Thiết lập lưới tìm kiếm**:
      - Xác định các siêu tham số như `k`, `min_k`, `sim_options` (phương pháp đo độ tương đồng: `pearson_baseline`, `cosine`, …).
    - **Tìm kiếm với GridSearchCV**:
      - Tìm bộ siêu tham số tối ưu dựa trên điểm RMSE thấp nhất.
      - In ra bộ siêu tham số tốt nhất và điểm RMSE tương ứng.
    """)
    st.image("Kếtquảtốiưuhóasiêuthamsố.png", caption="Kết quả tối ưu hóa siêu tham số", use_container_width=True)

    # 5. Áp dụng mô hình tối ưu
    st.markdown("##### 5. Áp dụng mô hình tối ưu")
    st.markdown("""
    - **Huấn luyện mô hình**:
      - Sử dụng mô hình KNNBaseline với các siêu tham số đã được tối ưu hóa.
      - Tách tập dữ liệu thành `trainset` và `testset`.
      - Huấn luyện mô hình trên `trainset`.
    - **Dự đoán và đánh giá**:
      - Thực hiện dự đoán trên `testset`.
      - Tính toán các chỉ số RMSE và MAE trên tập kiểm tra.
      - Hiển thị một số dự đoán mẫu để kiểm tra.
    """)
    st.image("Kếtquảtrêntậpkiểmtra.png", caption="Kết quả trên tập kiểm tra", use_container_width=True)

    # 6. Gợi ý sản phẩm
    st.markdown("##### 6. Gợi ý sản phẩm")
    st.markdown("""
    - **Lấy danh sách sản phẩm đã đánh giá**:
      - Xác định các sản phẩm đã được khách hàng cụ thể đánh giá và sắp xếp chúng theo `so_sao`.
    - **Dự đoán cho sản phẩm mới**:
      - Dự đoán điểm số cho các sản phẩm chưa được khách hàng đánh giá.
      - Lọc các sản phẩm có điểm dự đoán cao hơn ngưỡng tối thiểu (`min_rating = 4`).
      - Sắp xếp và chọn ra top 6 sản phẩm có điểm dự đoán cao nhất để gợi ý.
    - **Hiển thị kết quả**:
      - Kết hợp với thông tin sản phẩm và khách hàng để tạo báo cáo chi tiết.
    """)

    # 7. Lưu trữ mô hình
    st.markdown("##### 7. Lưu trữ mô hình")
    st.markdown("""
    - Sử dụng thư viện `joblib` để lưu lại mô hình đã huấn luyện (`KNNBaseline`).
    """)

    # Tổng kết
    st.markdown("""
    ### Kết quả từ quy trình trên:
    - Qua các thử nghiệm và đánh giá, nhận thấy rằng thuật toán tốt nhất là KNNBaseline với RMSE thấp nhất sau khi tối ưu hóa => Quyết định chọn mô hình để sử dụng cho hệ thống gợi ý của cửa hàng Hasaki.
    - **Các sản phẩm được gợi ý**: Dựa trên điểm dự đoán cao nhất từ mô hình.
    """)

    # Thuật toán 2. Content-based filtering (Cosine Similarity)
    st.subheader("II. Content-based filtering (Cosine Similarity)")
    
    # Đọc dữ liệu sản phẩm
    data = pd.read_csv('san_pham_processed.csv')
    st.write("##### 1. Chuẩn bị dữ liệu")
    st.dataframe(data[["ma_san_pham", "ten_san_pham", "gia_ban", "Content", "diem_trung_binh"]].head(3))

    st.write("##### 2. Xử lý dữ liệu")
    st.markdown("""
    Hàm tiền xử lý văn bản cơ bản bao gồm:
    1. Loại bỏ kí tự số.
    2. Loại bỏ các kí tự đặc biệt.
    3. Loại bỏ các từ stop words.
    """)
    st.image("process_text.jpg", use_container_width=True)
    st.dataframe(data[["Content", "Content_wt"]].head(3))

    st.write("##### 3. Xây dựng ma trận Consine Similarity")
    st.markdown("""
    - Tạo ma trận  Cosine Simalarity của `1200 sản phẩm`.
    """)
    st.image("cosine_matrix.jpg", use_container_width=True)

    st.write("##### 4. Viết hàm gợi ý sản phẩm")
    st.markdown("""
    - Viết hàm đề xuất sản phẩm tương tự dựa trên ma trận  `Cosine Simalarity`.
    """)
    st.image("cosine_function.jpg", use_container_width=True)
    st.image("test_result.jpg",caption="Kết quả đề xuất sản phẩm mới",  use_container_width=True)       


elif choice == 'Recommend New Product':  
    # Đọc dữ liệu sản phẩm
    df_products = pd.read_csv('san_pham_processed.csv')    
    # Open and read file to cosine_sim_new
    with open('products_cosine_sim.pkl', 'rb') as f:
        cosine_sim_new = pickle.load(f)

    # Tải dữ liệu từ file CSV
    df = pd.read_csv('training_data.csv')
    san_pham_df = pd.read_csv('san_pham_df.csv')
    df_khach_hang = pd.read_csv('df_khach_hang.csv')
    # Tải mô hình đã huấn luyện
    model_save_path = "KNNBaseline_model(1).joblib"
    algo = joblib.load(model_save_path)

    ###### Giao diện Streamlit ######
    # Hình ảnh banner
    # Images to display
    images = [
        "https://media.hcdn.vn/hsk/1733466722homeintoyou612.jpg",
        "https://media.hcdn.vn/hsk/1733480371homerelocation71.jpg",
        "https://media.hcdn.vn/hsk/1733394042homehada0512.jpg",
        "https://media.hcdn.vn/hsk/17335501101733549968727-82231569.jpg",
        "https://media.hcdn.vn/hsk/1733466722homeintoyou612.jpg",
        "https://media.hcdn.vn/hsk/1731315128846x250.jpg",        
        "https://media.hcdn.vn/hsk/1733302154homecamp1212-0412.png",
        "https://media.hcdn.vn/hsk/1733221853homeduongam0312.jpg"
    ]
    # Custom HTML for the slider
    slider_html = f"""
    <div class="slider">
        {"".join(f'<img src="{img}" alt="Slide" class="slide">' for img in images)}
    </div>

    <style>
    .slider {{
        display: flex;
        overflow: hidden;
        width: 700px;
        height: 200px;
        position: relative;
        margin: 0 auto;
    }}
    .slide {{
        min-width: 100%;
        transition: transform 1s ease;
    }}
    </style>

    <script>
    let index = 0;
    const slides = document.querySelectorAll('.slide');

    function autoSlide() {{
        index++;
        if (index >= slides.length) {{
            index = 0;
        }}
        slides.forEach((slide, i) => {{
            slide.style.transform = `translateX(-${{index * 100}}%)`;
        }});
    }}

    setInterval(autoSlide, 3000); // Change slide every 3 seconds
    </script>
    """
    # Embed the HTML in Streamlit
    st.components.v1.html(slider_html, height=200)
    st.markdown("Chào mừng bạn đến với hệ thống đề xuất sản phẩm của cửa hàng **Hasaki**! 🎉🎉🎉")  

    ########################################################################################################
    # LỰA CHỌN THUẬT TOÁN GỢI Ý
    method = st.sidebar.radio("Chọn phương pháp gợi ý sản phẩm", ("Content-Based Filtering", "Collaborative Filtering"))

    st.sidebar.header("Thông tin khách hàng")
    # Lọc danh sách khách hàng đã để lại đánh giá
    customers_with_reviews = df[df['so_sao'] > 0]['ma_khach_hang'].unique()
    customers_with_reviews_df = df_khach_hang[df_khach_hang['ma_khach_hang'].isin(customers_with_reviews)]
    # Lấy danh sách 20 khách hàng cố định (ngẫu nhiên)
    fixed_customers = customers_with_reviews_df.sample(n=20, random_state=42)
    # Sidebar: Chọn khách hàng
    selected_customer_id = st.sidebar.selectbox(
        "Chọn khách hàng",
        fixed_customers['ma_khach_hang'].tolist(),
        format_func=lambda x: f"ID: {x} - {fixed_customers[fixed_customers['ma_khach_hang'] == x]['ho_ten'].values[0]}"
    )
    # Tự động điền mật khẩu
    password = "123"
    st.sidebar.text_input("Mật khẩu (mặc định)", password, type="password", disabled=True)
    # Kiểm tra đăng nhập
    def check_login(customer_id, password):
        if password == "123":
            customer_info = df_khach_hang[df_khach_hang['ma_khach_hang'] == customer_id]
            if not customer_info.empty:
                return customer_info.iloc[0]['ho_ten']  # Trả về họ tên khách hàng
        return None
    # Đăng nhập
    if 'customer_name' not in st.session_state:
        st.session_state.customer_name = None

    if st.sidebar.button("Đăng nhập"):
        customer_name = check_login(selected_customer_id, password)
        if customer_name:
            st.session_state.customer_name = customer_name
            st.success(f"Xin chào {customer_name}!")
        else:
            st.error("Đăng nhập không thành công.")

    ########################################################################################################
    ## HÀM ĐỀ XUẤT CONTENT-BASED (CONSINE SIMILARITY)
    def get_recommendations(df, ma_san_pham, cosine_sim, nums=5):
        # Get the index of the product that matches the ma_san_pham
        matching_indices = df.index[df['ma_san_pham'] == ma_san_pham].tolist()
        if not matching_indices:
            print(f"No product found with ID: {ma_san_pham}")
            return pd.DataFrame()  # Return an empty DataFrame if no match
        idx = matching_indices[0]

        # Get the pairwise similarity scores of all products with that product
        sim_scores = list(enumerate(cosine_sim[idx]))
        # Sort the products based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # Get the scores of the nums most similar products (Ignoring the product itself)
        sim_scores = sim_scores[1:nums+1]
        # Get the product indices
        product_indices = [i[0] for i in sim_scores]
        # Filter the recommendations to include only products with rating > 4.0
        recommendations = df.iloc[product_indices]
        filtered_recommendations = recommendations[recommendations['diem_trung_binh'] > 4.0]
        # Return the top n most similar products as a DataFrame
        return filtered_recommendations

    # Hiển thị đề xuất ra bảng
    def display_recommended_products(recommended_products, cols=5):
        for i in range(0, len(recommended_products), cols):
            cols = st.columns(cols)
            for j, col in enumerate(cols):
                if i + j < len(recommended_products):
                    product = recommended_products.iloc[i + j]
                    with col:   
                        st.write(product['ten_san_pham'])       
                        st.write('###### 💵 Giá bán:', product['gia_ban'], 'VNĐ')
                        st.write('###### ⭐ Rating:', product['diem_trung_binh'])     
                        expander = st.expander(f"Mô tả")
                        product_description = product['mo_ta']
                        truncated_description = ' '.join(product_description.split()[:100]) + '...'
                        expander.write(truncated_description)
                        expander.markdown("Nhấn vào mũi tên để thu nhỏ.")                

    ########################################################################################################
    ## HÀM ĐỀ XUẤT COLLABORATIVE (SURPRISE)
    # Hàm rút gọn mô tả sản phẩm
    def truncate_description(description, max_words=20):
        words = description.split()
        if len(words) > max_words:
            return ' '.join(words[:max_words]) + '...'
        return description

    def display_products_with_rows(products_df, title, items_per_row=3):
        st.subheader(title)        
        # Sắp xếp lại các sản phẩm thành danh sách của các hàng
        rows = [products_df.iloc[i:i+items_per_row] for i in range(0, len(products_df), items_per_row)]        
        for row in rows:
            cols = st.columns(len(row))  # Tạo số cột tương ứng với số sản phẩm trong hàng
            for index, (col, (_, product)) in enumerate(zip(cols, row.iterrows())):
                with col:
                    st.markdown(f"### {product['ten_san_pham']}")
                    st.write(f"**Giá bán:** {product['gia_ban']} VNĐ")
                    st.write(f"**Điểm dự đoán:** {product['EstimateScore']}")
                    # Sử dụng st.expander để hiển thị mô tả sản phẩm
                    with st.expander("Xem mô tả sản phẩm", expanded=False):
                        truncated_description = truncate_description(product['mo_ta'])
                        st.write(f"**Mô tả đầy đủ:** {truncated_description}")

    # Hàm gợi ý sản phẩm Collaborative Filtering
    def recommend_products_for_collaborative(customer_id, df, algo, san_pham_df, df_khach_hang, top_n=6, min_rating=4):
        customer_purchased = df[df['ma_khach_hang'] == int(customer_id)][['ma_san_pham', 'so_sao']]
        if customer_purchased.empty:
            st.warning("Khách hàng này chưa mua sản phẩm nào.")
            show_highly_rated_products(san_pham_df, top_n=top_n)
            return None

        all_product_ids = df['ma_san_pham'].unique()
        products_to_predict = [product_id for product_id in all_product_ids if product_id not in customer_purchased['ma_san_pham'].tolist()]

        predictions_for_customer = []
        for product_id in products_to_predict:
            prediction = algo.predict(customer_id, product_id)
            predictions_for_customer.append(prediction)

        predictions_df = pd.DataFrame(
            [(pred.iid, pred.est) for pred in predictions_for_customer],
            columns=['ma_san_pham', 'EstimateScore']
        )
        filtered_predictions = predictions_df[predictions_df['EstimateScore'] >= min_rating] 
        if filtered_predictions.empty:
            st.warning("Không có sản phẩm gợi ý phù hợp.")
            return None

        merged_df = filtered_predictions.merge(san_pham_df, on='ma_san_pham', how='left')
        final_recommendations = merged_df[merged_df['diem_trung_binh'] >= min_rating]
        final_recommendations = final_recommendations[['ma_san_pham', 'ten_san_pham', 'mo_ta', 'gia_ban', 'EstimateScore']]
        final_recommendations['ma_khach_hang'] = int(customer_id)
        final_recommendations = final_recommendations.merge(df_khach_hang[['ma_khach_hang', 'ho_ten']], on='ma_khach_hang', how='left')
        final_recommendations = final_recommendations.sort_values(by='EstimateScore', ascending=False).head(top_n)
        return final_recommendations
    
    ########################################################################################################   
    # THUẬT TOÁN 1: Content-Based Filtering
    if method == "Content-Based Filtering": 
        # Lấy 15 sản phẩm
        random_products = df_products.head(n=20)
        # print(random_products)

        # Giữ trạng thái hạn chế nhảy lung tung
        st.session_state.random_products = random_products
        # Kiểm tra xem 'selected_ma_san_pham' đã có trong session_state hay chưa
        if 'selected_ma_san_pham' not in st.session_state:
            # Nếu chưa có, thiết lập giá trị mặc định là None hoặc ID sản phẩm đầu tiên
            st.session_state.selected_ma_san_pham = None

        # Theo cách cho người dùng chọn sản phẩm từ dropdown
        # Tạo một tuple cho mỗi sản phẩm, trong đó phần tử đầu là tên và phần tử thứ hai là ID
        product_options = [(row['ten_san_pham'], row['ma_san_pham']) for index, row in st.session_state.random_products.iterrows()]
        st.session_state.random_products

        # Người dùng chọn 1 sản phẩm trong list random hoặc tự nhập vào mã sản phẩm
        # Tạo một dropdown với options là các tuple này
        selected_product = st.selectbox(
            "Chọn sản phẩm trong Dropbox",
            options=product_options,            
            format_func=lambda x: x[0]  # Hiển thị tên sản phẩm
        )

        # Display the selected product
        st.write("Bạn đã chọn:", selected_product)

        # Cập nhật session_state dựa trên lựa chọn hiện tại
        st.session_state.selected_ma_san_pham = selected_product[1]

        if st.session_state.selected_ma_san_pham:
            st.write("Mã sản phẩm: ", st.session_state.selected_ma_san_pham)
            # Hiển thị thông tin sản phẩm được chọn
            selected_product = df_products[df_products['ma_san_pham'] == st.session_state.selected_ma_san_pham]
            if not selected_product.empty:
                st.write('#### Bạn vừa chọn:')
                st.write('### ', selected_product['ten_san_pham'].values[0])     

                original_price = 500000
                # Hiển thị giá gốc và giá bán
                st.markdown(
                    f"""
                    <p style="font-size:16px;">
                        💵Giá gốc: <span style="color: red; text-decoration: line-through;">{selected_product['gia_goc'].values[0]:,} VNĐ</span>
                    </p>
                    <p style="font-size:16px; font-weight: bold; color: green;">
                        💵Giá bán: {selected_product['gia_ban'].values[0]:,} VNĐ
                    </p>
                    """,
                    unsafe_allow_html=True,
                )

                st.write('###### ⭐ Rating:', selected_product['diem_trung_binh'].values[0])

                product_description = selected_product['mo_ta'].values[0]
                truncated_description = ' '.join(product_description.split()[:100])
                # st.write('##### **Thông tin sản phẩm:**')
                st.markdown("""##### ***Thông tin sản phẩm:*** """)
                st.write(truncated_description, '...')

                # Đề xuất sản phẩm liên quan
                st.markdown('##### ***Các sản phẩm liên quan:***')
                recommendations = get_recommendations(df_products, st.session_state.selected_ma_san_pham, cosine_sim=cosine_sim_new, nums=3) 
                display_recommended_products(recommendations, cols=3)
            else:
                st.write(f"Không tìm thấy sản phẩm với ID: {st.session_state.selected_ma_san_pham}")


    ########################################################################################################
    # THUẬT TOÁN 2: Collaborative Filtering
    elif method == "Collaborative Filtering" and st.session_state.customer_name: 
        if st.sidebar.button("Gợi ý sản phẩm"):
            recommended_products = recommend_products_for_collaborative(selected_customer_id, df, algo, san_pham_df, df_khach_hang)
            if recommended_products is not None:
                display_products_with_rows(recommended_products, f"🔸Sản phẩm đề xuất cho khách hàng: {st.session_state.customer_name}")

            # Hiển thị sản phẩm đã được đánh giá bởi khách hàng
            customer_purchased = df[df['ma_khach_hang'] == int(selected_customer_id)][['ma_san_pham', 'so_sao']]
            if not customer_purchased.empty:
                customer_purchased = customer_purchased.merge(
                    san_pham_df[['ma_san_pham', 'ten_san_pham', 'diem_trung_binh', 'gia_ban', 'mo_ta']],
                    on='ma_san_pham',
                    how='left'
                ).sort_values(by='so_sao', ascending=False)
                st.subheader(f"🔸 Sản phẩm đã được đánh giá bởi khách hàng: {st.session_state.customer_name}")
                st.write(customer_purchased[['ma_san_pham', 'ten_san_pham', 'so_sao']])
            else:
                st.warning("Khách hàng này chưa mua sản phẩm nào.")


    

    