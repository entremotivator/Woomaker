import streamlit as st
import openai
import requests
import json
import base64
import pandas as pd
from typing import Dict, Any, List, Optional
from uuid import uuid4
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

st.set_page_config(page_title="WooCommerce Product Manager", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better UI
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 1rem;
}
.product-card {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 1rem;
    background: #f9f9f9;
}
.success-alert {
    background-color: #d4edda;
    border-color: #c3e6cb;
    color: #155724;
    padding: 0.75rem 1.25rem;
    margin-bottom: 1rem;
    border: 1px solid transparent;
    border-radius: 0.25rem;
}
.error-alert {
    background-color: #f8d7da;
    border-color: #f5c6cb;
    color: #721c24;
    padding: 0.75rem 1.25rem;
    margin-bottom: 1rem;
    border: 1px solid transparent;
    border-radius: 0.25rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Helper Functions ----------------
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_woo_products(url: str, ck: str, cs: str, per_page: int = 20, page: int = 1, search: str = "") -> tuple:
    """Fetch products from WooCommerce store"""
    endpoint = f"{url.rstrip('/')}/wp-json/wc/v3/products"
    params = {
        'per_page': per_page,
        'page': page,
        'orderby': 'date',
        'order': 'desc'
    }
    if search:
        params['search'] = search
    
    try:
        response = requests.get(endpoint, auth=(ck, cs), params=params)
        products = response.json() if response.status_code == 200 else []
        total_products = int(response.headers.get('X-WP-Total', 0))
        total_pages = int(response.headers.get('X-WP-TotalPages', 0))
        return products, total_products, total_pages
    except Exception as e:
        st.error(f"Error fetching products: {e}")
        return [], 0, 0

@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_woo_orders(url: str, ck: str, cs: str, per_page: int = 50) -> list:
    """Fetch recent orders from WooCommerce store"""
    endpoint = f"{url.rstrip('/')}/wp-json/wc/v3/orders"
    params = {'per_page': per_page, 'orderby': 'date', 'order': 'desc'}
    
    try:
        response = requests.get(endpoint, auth=(ck, cs), params=params)
        return response.json() if response.status_code == 200 else []
    except Exception as e:
        st.error(f"Error fetching orders: {e}")
        return []

def update_woo_product(url: str, ck: str, cs: str, product_id: int, data: dict) -> tuple:
    """Update a product in WooCommerce"""
    endpoint = f"{url.rstrip('/')}/wp-json/wc/v3/products/{product_id}"
    try:
        response = requests.put(endpoint, auth=(ck, cs), json=data)
        return response.status_code, response.json()
    except Exception as e:
        return 400, {"message": str(e)}

def delete_woo_product(url: str, ck: str, cs: str, product_id: int) -> tuple:
    """Delete a product from WooCommerce"""
    endpoint = f"{url.rstrip('/')}/wp-json/wc/v3/products/{product_id}"
    try:
        response = requests.delete(endpoint, auth=(ck, cs), params={'force': True})
        return response.status_code, response.json()
    except Exception as e:
        return 400, {"message": str(e)}

def calculate_metrics(orders: list, products: list) -> dict:
    """Calculate store metrics"""
    if not orders:
        return {
            'total_revenue': 0,
            'total_orders': 0,
            'avg_order_value': 0,
            'top_products': [],
            'revenue_trend': [],
            'order_status_dist': {}
        }
    
    total_revenue = sum(float(order.get('total', 0)) for order in orders)
    total_orders = len(orders)
    avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
    
    # Product sales analysis
    product_sales = {}
    for order in orders:
        for item in order.get('line_items', []):
            product_id = item.get('product_id')
            quantity = item.get('quantity', 0)
            if product_id in product_sales:
                product_sales[product_id] += quantity
            else:
                product_sales[product_id] = quantity
    
    # Get top products
    top_product_ids = sorted(product_sales.items(), key=lambda x: x[1], reverse=True)[:5]
    top_products = []
    for product_id, sales in top_product_ids:
        product = next((p for p in products if p.get('id') == product_id), None)
        if product:
            top_products.append({
                'name': product.get('name', f'Product {product_id}'),
                'sales': sales,
                'revenue': sales * float(product.get('price', 0))
            })
    
    # Revenue trend (last 30 days)
    revenue_trend = []
    for i in range(30):
        date = datetime.now() - timedelta(days=i)
        day_revenue = sum(
            float(order.get('total', 0)) 
            for order in orders 
            if order.get('date_created', '').startswith(date.strftime('%Y-%m-%d'))
        )
        revenue_trend.append({'date': date.strftime('%Y-%m-%d'), 'revenue': day_revenue})
    
    # Order status distribution
    order_status_dist = {}
    for order in orders:
        status = order.get('status', 'unknown')
        order_status_dist[status] = order_status_dist.get(status, 0) + 1
    
    return {
        'total_revenue': total_revenue,
        'total_orders': total_orders,
        'avg_order_value': avg_order_value,
        'top_products': top_products,
        'revenue_trend': revenue_trend,
        'order_status_dist': order_status_dist
    }

# ---------------- Sidebar Configuration ----------------
st.sidebar.title("🔑 API Keys & Settings")

# API Keys
openai_api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    placeholder="sk-...",
    help="Enter your OpenAI API key (never stored)."
)

# WooCommerce Configuration
st.sidebar.markdown("### WooCommerce Connection")
woo_url = st.sidebar.text_input("Store URL", placeholder="https://yourstore.com")
woo_ck = st.sidebar.text_input("Consumer Key", type="password", placeholder="ck_...")
woo_cs = st.sidebar.text_input("Consumer Secret", type="password", placeholder="cs_...")

# Test connection
woo_connected = False
if woo_url and woo_ck and woo_cs:
    if st.sidebar.button("🔗 Test Connection"):
        try:
            test_response = requests.get(f"{woo_url.rstrip('/')}/wp-json/wc/v3/products", 
                                       auth=(woo_ck, woo_cs), params={'per_page': 1})
            if test_response.status_code == 200:
                st.sidebar.success("✅ Connection successful!")
                woo_connected = True
            else:
                st.sidebar.error(f"❌ Connection failed: {test_response.status_code}")
        except Exception as e:
            st.sidebar.error(f"❌ Connection error: {e}")

# Generation Settings
st.sidebar.markdown("### AI Settings")
model = st.sidebar.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], index=0)
image_size = st.sidebar.selectbox("Image size", ["512x512", "1024x1024"], index=1)

if openai_api_key:
    openai.api_key = openai_api_key

# ---------------- Main Navigation ----------------
st.title("🛒 WooCommerce Product Manager")
st.caption("Complete AI-powered product generation and store management system")

# Navigation tabs
tab1, tab2, tab3, tab4 = st.tabs(["🚀 Generate Products", "📊 Store Dashboard", "🛠️ Manage Products", "📈 Analytics"])

# ---------------- TAB 1: Product Generation ----------------
with tab1:
    st.header("AI Product Generator")
    
    if not openai_api_key:
        st.warning("👉 Enter your OpenAI API key in the sidebar to start generating products.")
    else:
        with st.form("product_form"):
            st.subheader("Product Brief")
            
            # Pre-made templates
            template = st.selectbox("Quick Templates", [
                "Custom",
                "Eco-friendly water bottle",
                "Luxury skincare product",
                "Tech gadget accessory",
                "Fitness equipment",
                "Home decor item",
                "Fashion accessory",
                "Kitchen appliance"
            ])
            
            templates = {
                "Eco-friendly water bottle": "Eco-friendly stainless steel water bottle, 24oz, double-wall insulated, keeps drinks cold for 24h and hot for 12h. Leak-proof lid. Colors: black, silver, blue. Target audience: fitness enthusiasts, travelers, office workers.",
                "Luxury skincare product": "Premium anti-aging serum with hyaluronic acid and vitamin C. 30ml bottle, suitable for all skin types. Reduces fine lines and improves skin texture. Target audience: adults 25-55 interested in skincare.",
                "Tech gadget accessory": "Wireless charging stand for smartphones, 15W fast charging, adjustable angle, LED indicator, compatible with iPhone and Android. Colors: black, white. Target: tech-savvy users.",
                "Fitness equipment": "Resistance bands set with 5 different resistance levels, door anchor, handles, and workout guide. Portable home gym equipment. Target: fitness enthusiasts, home workout users.",
                "Home decor item": "Modern minimalist table lamp with touch control, dimmable LED, wireless charging base for phones. Available in white and black. Target: modern home decorators.",
                "Fashion accessory": "Handcrafted leather crossbody bag with adjustable strap, multiple compartments, genuine leather. Colors: brown, black, tan. Target: fashion-conscious women 20-45.",
                "Kitchen appliance": "Electric milk frother with hot and cold frothing modes, stainless steel, 240ml capacity, automatic shut-off. Perfect for coffee lovers and home baristas."
            }
            
            product_brief = st.text_area(
                "Describe your product:",
                height=140,
                value=templates.get(template, "")
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                tone = st.selectbox("Writing tone", ["friendly", "luxury", "professional", "funny", "technical", "persuasive"])
            with col2:
                language = st.selectbox("Language", ["English", "Spanish", "French", "German", "Italian", "Portuguese"])
            with col3:
                target_audience = st.selectbox("Target Audience", ["General", "Luxury buyers", "Budget-conscious", "Tech enthusiasts", "Fitness enthusiasts", "Eco-conscious"])
            
            seo_keywords = st.text_input("SEO Keywords (comma separated)", value="")

            st.subheader("Product Configuration")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                price_usd = st.number_input("Base Price (USD)", min_value=0.0, value=25.99, step=0.01)
            with col2:
                sku_prefix = st.text_input("SKU Prefix", value="PROD")
            with col3:
                stock_qty = st.number_input("Stock Quantity", min_value=0, value=50, step=1)
            with col4:
                sale_price = st.number_input("Sale Price (optional)", min_value=0.0, value=0.0, step=0.01)

            col1, col2, col3 = st.columns(3)
            with col1:
                generate_variations = st.checkbox("Generate Variations (colors/sizes)", value=True)
            with col2:
                generate_images = st.checkbox("Generate AI Images", value=True)
            with col3:
                auto_publish = st.checkbox("Auto-publish to WooCommerce", value=False, disabled=not (woo_url and woo_ck and woo_cs))
            
            n_products = st.slider("Number of Products to Generate", 1, 10, 1)
            
            submitted = st.form_submit_button("🚀 Generate Products", use_container_width=True)

        # Enhanced prompt builder
        def build_enhanced_prompt(brief, tone, language, price, sku_prefix, stock, n, variations, seo_kw, target_audience, sale_price):
            variation_text = "Include color and size variations with different SKUs" if variations else "Single product only"
            sale_text = f"Include sale_price: '{sale_price}'" if sale_price > 0 else "No sale price"
            
            return f"""
You are an expert ecommerce copywriter and product data engineer specializing in WooCommerce.
Create {n} high-converting, WooCommerce-ready product JSON objects.

REQUIREMENTS:
- Language: {language}
- Writing Tone: {tone} (make copy compelling and {tone})
- Target Audience: {target_audience}
- Base Price: ${price} USD
- SKU Prefix: {sku_prefix}
- Stock Quantity: {stock}
- Variations: {variation_text}
- SEO Keywords: {seo_kw}
- Sale Price: {sale_text}

PRODUCT BRIEF: {brief}

Each product JSON must include ALL these fields:
- id: (unique uuid4)
- name: (compelling, SEO-optimized title 50-60 chars)
- slug: (url-friendly version of name)
- short_description: (2-3 sentences with HTML, highlighting key benefits)
- description: (detailed HTML copy with <h3> subheadings, <ul><li> features, persuasive benefits, call-to-action)
- sku: (unique: {sku_prefix}_[random])
- regular_price: (string format)
- sale_price: (string format or empty)
- manage_stock: true
- stock_quantity: (integer)
- stock_status: "instock"
- categories: [list of {{"name": "category"}}]
- tags: [list of relevant tag strings]
- attributes: [list with name, options, variation boolean]
- images: [list of {{"prompt": "detailed AI image prompt for product photography"}}]
- meta_data: {{"meta_title": "SEO title", "meta_description": "SEO meta description"}}
- weight: (string, reasonable product weight)
- dimensions: {{"length": "x", "width": "y", "height": "z"}}

IMPORTANT: Return ONLY valid JSON array, no explanations or markdown.
"""

        # Generation functions
        def generate_products_json(prompt_text, model):
            messages = [
                {"role": "system", "content": "You are a WooCommerce product data expert. Return only valid JSON arrays with complete product data."},
                {"role": "user", "content": prompt_text},
            ]
            resp = openai.ChatCompletion.create(model=model, messages=messages, temperature=0.7, max_tokens=4000)
            text = resp["choices"][0]["message"]["content"].strip()
            
            # Clean up JSON formatting
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            
            return json.loads(text)

        def generate_image(prompt, size="1024x1024"):
            try:
                result = openai.Image.create(prompt=f"Professional product photography: {prompt}. White background, studio lighting, high quality, commercial photography style.", n=1, size=size)
                if "url" in result["data"][0]:
                    return result["data"][0]["url"]
                b64 = result["data"][0].get("b64_json")
                return "data:image/png;base64," + b64
            except Exception as e:
                st.error(f"Image generation failed: {e}")
                return None

        def woo_publish(product, url, ck, cs):
            endpoint = f"{url.rstrip('/')}/wp-json/wc/v3/products"
            try:
                r = requests.post(endpoint, auth=(ck, cs), json=product, timeout=30)
                return r.status_code, r.json()
            except Exception as e:
                return 400, {"message": str(e)}

        # Product generation workflow
        if submitted:
            if not product_brief.strip():
                st.error("Please provide a product description.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Generate products
                status_text.text("🤖 Generating product data...")
                progress_bar.progress(20)
                
                prompt_text = build_enhanced_prompt(
                    product_brief, tone, language, price_usd, sku_prefix, 
                    stock_qty, n_products, generate_variations, seo_keywords, 
                    target_audience, sale_price
                )
                
                try:
                    products = generate_products_json(prompt_text, model)
                    progress_bar.progress(40)
                    status_text.text(f"✅ Generated {len(products)} products!")
                    
                    # Display and process products
                    for idx, product in enumerate(products):
                        st.markdown("---")
                        
                        # Product header
                        col1, col2, col3 = st.columns([2, 1, 1])
                        with col1:
                            st.subheader(f"🛍️ {product.get('name', 'Unnamed Product')}")
                        with col2:
                            st.metric("Price", f"${product.get('regular_price', '0')}")
                        with col3:
                            st.metric("Stock", f"{product.get('stock_quantity', 0)} units")
                        
                        # Product details
                        tab_desc, tab_json, tab_actions = st.tabs(["📝 Description", "📋 JSON Data", "⚡ Actions"])
                        
                        with tab_desc:
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.markdown("**Short Description**")
                                st.markdown(product.get("short_description", ""), unsafe_allow_html=True)
                                
                                st.markdown("**Full Description**")
                                st.markdown(product.get("description", ""), unsafe_allow_html=True)
                                
                                # SEO Info
                                if product.get("meta_data"):
                                    st.markdown("**SEO Meta Data**")
                                    st.info(f"**Title:** {product['meta_data'].get('meta_title', 'N/A')}\n\n**Description:** {product['meta_data'].get('meta_description', 'N/A')}")
                            
                            with col2:
                                st.markdown("**Product Details**")
                                details = {
                                    "SKU": product.get("sku", "N/A"),
                                    "Regular Price": f"${product.get('regular_price', '0')}",
                                    "Sale Price": f"${product.get('sale_price', 'N/A')}" if product.get('sale_price') else "No Sale",
                                    "Stock": f"{product.get('stock_quantity', 0)} units",
                                    "Weight": f"{product.get('weight', 'N/A')}",
                                    "Categories": ", ".join([cat.get('name', '') for cat in product.get('categories', [])]),
                                    "Tags": ", ".join(product.get('tags', []))
                                }
                                
                                for key, value in details.items():
                                    st.text(f"{key}: {value}")
                        
                        with tab_json:
                            st.json(product)
                            
                            # Download options
                            col1, col2 = st.columns(2)
                            with col1:
                                st.download_button(
                                    "📥 Download JSON",
                                    json.dumps(product, indent=2).encode('utf-8'),
                                    file_name=f"{product.get('sku', 'product')}.json",
                                    mime="application/json",
                                    key=f"json_{idx}"
                                )
                            with col2:
                                # CSV format
                                df = pd.json_normalize(product)
                                st.download_button(
                                    "📥 Download CSV",
                                    df.to_csv(index=False).encode('utf-8'),
                                    file_name=f"{product.get('sku', 'product')}.csv",
                                    mime="text/csv",
                                    key=f"csv_{idx}"
                                )
                        
                        with tab_actions:
                            col1, col2 = st.columns(2)
                            
                            # Image generation
                            with col1:
                                st.markdown("**🎨 Generate Image**")
                                if generate_images and product.get("images"):
                                    img_prompt = product["images"][0].get("prompt", f"Product photo of {product.get('name')}")
                                    
                                    if st.button(f"Generate Image", key=f"img_{idx}"):
                                        with st.spinner("Generating image..."):
                                            img_url = generate_image(img_prompt, image_size)
                                            if img_url:
                                                st.image(img_url, caption=img_prompt, width=300)
                                            else:
                                                st.error("Failed to generate image")
                                    
                                    st.caption(f"Prompt: {img_prompt[:100]}...")
                            
                            # WooCommerce publishing
                            with col2:
                                st.markdown("**🚀 Publish to Store**")
                                if woo_url and woo_ck and woo_cs:
                                    if auto_publish:
                                        status_text.text(f"📤 Publishing {product.get('sku')} to WooCommerce...")
                                        code, resp = woo_publish(product, woo_url, woo_ck, woo_cs)
                                        if code in (200, 201):
                                            st.success(f"✅ Published! WooCommerce ID: {resp.get('id')}")
                                        else:
                                            st.error(f"❌ Publishing failed: {resp.get('message', 'Unknown error')}")
                                    else:
                                        if st.button(f"📤 Publish to WooCommerce", key=f"pub_{idx}"):
                                            with st.spinner("Publishing..."):
                                                code, resp = woo_publish(product, woo_url, woo_ck, woo_cs)
                                                if code in (200, 201):
                                                    st.success(f"✅ Published! WooCommerce ID: {resp.get('id')}")
                                                    st.balloons()
                                                else:
                                                    st.error(f"❌ Publishing failed: {resp.get('message', 'Unknown error')}")
                                else:
                                    st.warning("Configure WooCommerce settings in sidebar to publish")
                        
                        progress_bar.progress(40 + (idx + 1) * (60 // len(products)))
                    
                    # Bulk actions
                    st.markdown("---")
                    st.subheader("📦 Bulk Actions")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        all_json = json.dumps(products, indent=2).encode('utf-8')
                        st.download_button(
                            "📥 Download All JSON",
                            all_json,
                            file_name="generated_products.json",
                            mime="application/json"
                        )
                    
                    with col2:
                        # Combined CSV
                        df_all = pd.json_normalize(products)
                        st.download_button(
                            "📥 Download All CSV",
                            df_all.to_csv(index=False).encode('utf-8'),
                            file_name="generated_products.csv",
                            mime="text/csv"
                        )
                    
                    with col3:
                        if woo_url and woo_ck and woo_cs and not auto_publish:
                            if st.button("🚀 Publish All to WooCommerce"):
                                published_count = 0
                                for product in products:
                                    code, resp = woo_publish(product, woo_url, woo_ck, woo_cs)
                                    if code in (200, 201):
                                        published_count += 1
                                st.success(f"✅ Published {published_count}/{len(products)} products successfully!")
                    
                    progress_bar.progress(100)
                    status_text.text("🎉 Generation complete!")
                    
                except json.JSONDecodeError as e:
                    st.error(f"❌ Error parsing generated JSON: {e}")
                    st.error("The AI may have returned invalid JSON. Please try again.")
                except Exception as e:
                    st.error(f"❌ Generation failed: {e}")

# ---------------- TAB 2: Store Dashboard ----------------
with tab2:
    st.header("📊 Store Dashboard")
    
    if not (woo_url and woo_ck and woo_cs):
        st.warning("👉 Configure WooCommerce settings in the sidebar to view store dashboard.")
    else:
        # Refresh button
        if st.button("🔄 Refresh Dashboard Data", key="refresh_dashboard"):
            st.cache_data.clear()
            st.rerun()
        
        with st.spinner("Loading store data..."):
            # Fetch data
            products, total_products, total_pages = get_woo_products(woo_url, woo_ck, woo_cs, per_page=100)
            orders = get_woo_orders(woo_url, woo_ck, woo_cs, per_page=100)
            metrics = calculate_metrics(orders, products)
        
        if products or orders:
            # Key Metrics
            st.subheader("📈 Key Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <h3>$""" + f"{metrics['total_revenue']:,.2f}" + """</h3>
                    <p>Total Revenue</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{metrics['total_orders']}</h3>
                    <p>Total Orders</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{total_products}</h3>
                    <p>Total Products</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>${metrics['avg_order_value']:,.2f}</h3>
                    <p>Avg Order Value</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                if metrics['revenue_trend']:
                    st.subheader("📊 Revenue Trend (30 Days)")
                    trend_df = pd.DataFrame(metrics['revenue_trend'])
                    trend_df['date'] = pd.to_datetime(trend_df['date'])
                    
                    fig = px.line(trend_df, x='date', y='revenue', 
                                title="Daily Revenue",
                                labels={'revenue': 'Revenue ($)', 'date': 'Date'})
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if metrics['order_status_dist']:
                    st.subheader("📋 Order Status Distribution")
                    status_df = pd.DataFrame(list(metrics['order_status_dist'].items()), 
                                           columns=['Status', 'Count'])
                    
                    fig = px.pie(status_df, values='Count', names='Status', 
                               title="Order Status Breakdown")
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Top Products
            if metrics['top_products']:
                st.subheader("🏆 Top Selling Products")
                top_products_df = pd.DataFrame(metrics['top_products'])
                
                for idx, product in enumerate(metrics['top_products'][:5]):
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.write(f"**{idx+1}. {product['name']}**")
                    with col2:
                        st.write(f"{product['sales']} sold")
                    with col3:
                        st.write(f"${product['revenue']:,.2f}")
            
            # Recent Orders
            if orders:
                st.subheader("🛍️ Recent Orders")
                recent_orders = orders[:10]  # Show last 10 orders
                
                orders_data = []
                for order in recent_orders:
                    orders_data.append({
                        'Order ID': order.get('id', 'N/A'),
                        'Customer': f"{order.get('billing', {}).get('first_name', '')} {order.get('billing', {}).get('last_name', '')}".strip() or 'Guest',
                        'Total': f"${float(order.get('total', 0)):,.2f}",
                        'Status': order.get('status', 'unknown').title(),
                        'Date': order.get('date_created', '')[:10] if order.get('date_created') else 'N/A'
                    })
                
                if orders_data:
                    st.dataframe(pd.DataFrame(orders_data), use_container_width=True)
        else:
            st.info("No store data available. Make sure your WooCommerce store has products and orders.")

# ---------------- TAB 3: Manage Products ----------------
with tab3:
    st.header("🛠️ Product Management")
    
    if not (woo_url and woo_ck and woo_cs):
        st.warning("👉 Configure WooCommerce settings in the sidebar to manage products.")
    else:
        # Search and filters
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            search_term = st.text_input("🔍 Search products", placeholder="Enter product name or SKU...")
        with col2:
            products_per_page = st.selectbox("Products per page", [10, 20, 50, 100], index=1)
        with col3:
            if st.button("🔄 Refresh Products"):
                st.cache_data.clear()
                st.rerun()
        
        # Fetch products
        page = st.session_state.get('current_page', 1)
        
        with st.spinner("Loading products..."):
            products, total_products, total_pages = get_woo_products(
                woo_url, woo_ck, woo_cs, 
                per_page=products_per_page, 
                page=page, 
                search=search_term
            )
        
        if products:
            # Pagination info
            st.caption(f"Showing {len(products)} of {total_products} products (Page {page} of {total_pages})")
            
            # Bulk actions
            st.subheader("🔧 Bulk Actions")
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                bulk_action = st.selectbox("Select Action", [
                    "None",
                    "Update Stock Status",
                    "Apply Discount",
                    "Update Category",
                    "Export Selected"
                ])
            
            with col2:
                if bulk_action == "Update Stock Status":
                    new_stock_status = st.selectbox("New Status", ["instock", "outofstock", "onbackorder"])
                elif bulk_action == "Apply Discount":
                    discount_percent = st.number_input("Discount %", min_value=0, max_value=90, value=10)
                elif bulk_action == "Update Category":
                    new_category = st.text_input("Category Name")
            
            # Product list with management options
            selected_products = []
            
            for idx, product in enumerate(products):
                with st.expander(f"📦 {product.get('name', 'Unnamed Product')} (ID: {product.get('id')})"):
                    # Selection checkbox
                    selected = st.checkbox(f"Select", key=f"select_{product.get('id')}")
                    if selected:
                        selected_products.append(product.get('id'))
                    
                    # Product info
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.markdown(f"**SKU:** {product.get('sku', 'N/A')}")
                        st.markdown(f"**Status:** {product.get('status', 'unknown').title()}")
                        st.markdown(f"**Type:** {product.get('type', 'simple').title()}")
                        
                        # Short description (editable)
                        new_short_desc = st.text_area(
                            "Short Description",
                            value=product.get('short_description', ''),
                            height=60,
                            key=f"short_desc_{product.get('id')}"
                        )
                    
                    with col2:
                        # Price editing
                        current_price = float(product.get('regular_price', 0))
                        new_regular_price = st.number_input(
                            "Regular Price",
                            min_value=0.0,
                            value=current_price,
                            step=0.01,
                            key=f"price_{product.get('id')}"
                        )
                        
                        current_sale_price = float(product.get('sale_price', 0)) if product.get('sale_price') else 0.0
                        new_sale_price = st.number_input(
                            "Sale Price",
                            min_value=0.0,
                            value=current_sale_price,
                            step=0.01,
                            key=f"sale_price_{product.get('id')}"
                        )
                    
                    with col3:
                        # Stock management
                        current_stock = product.get('stock_quantity', 0)
                        new_stock_qty = st.number_input(
                            "Stock Quantity",
                            min_value=0,
                            value=current_stock if current_stock else 0,
                            key=f"stock_{product.get('id')}"
                        )
                        
                        new_stock_status = st.selectbox(
                            "Stock Status",
                            ["instock", "outofstock", "onbackorder"],
                            index=["instock", "outofstock", "onbackorder"].index(product.get('stock_status', 'instock')),
                            key=f"stock_status_{product.get('id')}"
                        )
                    
                    # Categories and tags
                    st.markdown("**Categories & Tags**")
                    current_categories = [cat.get('name') for cat in product.get('categories', [])]
                    new_categories = st.text_input(
                        "Categories (comma separated)",
                        value=', '.join(current_categories),
                        key=f"categories_{product.get('id')}"
                    )
                    
                    current_tags = product.get('tags', [])
                    if isinstance(current_tags[0], dict) if current_tags else False:
                        current_tags = [tag.get('name') for tag in current_tags]
                    new_tags = st.text_input(
                        "Tags (comma separated)",
                        value=', '.join(current_tags) if current_tags else '',
                        key=f"tags_{product.get('id')}"
                    )
                    
                    # Action buttons
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button(f"💾 Update Product", key=f"update_{product.get('id')}"):
                            # Prepare update data
                            update_data = {}
                            
                            # Only include changed fields
                            if new_regular_price != current_price:
                                update_data['regular_price'] = str(new_regular_price)
                            
                            if new_sale_price != current_sale_price:
                                update_data['sale_price'] = str(new_sale_price) if new_sale_price > 0 else ""
                            
                            if new_stock_qty != current_stock:
                                update_data['stock_quantity'] = new_stock_qty
                            
                            if new_stock_status != product.get('stock_status'):
                                update_data['stock_status'] = new_stock_status
                            
                            if new_short_desc != product.get('short_description', ''):
                                update_data['short_description'] = new_short_desc
                            
                            if new_categories != ', '.join(current_categories):
                                update_data['categories'] = [{'name': cat.strip()} for cat in new_categories.split(',') if cat.strip()]
                            
                            if new_tags != ', '.join(current_tags):
                                update_data['tags'] = [{'name': tag.strip()} for tag in new_tags.split(',') if tag.strip()]
                            
                            if update_data:
                                with st.spinner("Updating product..."):
                                    code, response = update_woo_product(woo_url, woo_ck, woo_cs, product.get('id'), update_data)
                                    if code == 200:
                                        st.success("✅ Product updated successfully!")
                                        time.sleep(1)
                                        st.rerun()
                                    else:
                                        st.error(f"❌ Update failed: {response.get('message', 'Unknown error')}")
                            else:
                                st.info("No changes to update")
                    
                    with col2:
                        if st.button(f"👁️ View on Store", key=f"view_{product.get('id')}"):
                            store_url = f"{woo_url.rstrip('/')}/product/{product.get('slug', '')}"
                            st.markdown(f"[🔗 View Product]({store_url})")
                    
                    with col3:
                        if st.button(f"🗑️ Delete Product", key=f"delete_{product.get('id')}", type="secondary"):
                            if st.session_state.get(f"confirm_delete_{product.get('id')}", False):
                                with st.spinner("Deleting product..."):
                                    code, response = delete_woo_product(woo_url, woo_ck, woo_cs, product.get('id'))
                                    if code == 200:
                                        st.success("✅ Product deleted successfully!")
                                        time.sleep(1)
                                        st.rerun()
                                    else:
                                        st.error(f"❌ Delete failed: {response.get('message', 'Unknown error')}")
                            else:
                                st.warning("Click again to confirm deletion")
                                st.session_state[f"confirm_delete_{product.get('id')}"] = True
            
            # Execute bulk actions
            if bulk_action != "None" and selected_products:
                if st.button(f"Execute Bulk Action for {len(selected_products)} products"):
                    success_count = 0
                    
                    for product_id in selected_products:
                        update_data = {}
                        
                        if bulk_action == "Update Stock Status":
                            update_data = {'stock_status': new_stock_status}
                        elif bulk_action == "Apply Discount":
                            # Find the product to get current price
                            current_product = next((p for p in products if p.get('id') == product_id), None)
                            if current_product:
                                current_price = float(current_product.get('regular_price', 0))
                                discounted_price = current_price * (1 - discount_percent / 100)
                                update_data = {'sale_price': str(discounted_price)}
                        elif bulk_action == "Update Category":
                            if new_category:
                                update_data = {'categories': [{'name': new_category}]}
                        
                        if update_data:
                            code, response = update_woo_product(woo_url, woo_ck, woo_cs, product_id, update_data)
                            if code == 200:
                                success_count += 1
                    
                    st.success(f"✅ Bulk action completed successfully for {success_count}/{len(selected_products)} products!")
                    time.sleep(2)
                    st.rerun()
            
            # Pagination
            if total_pages > 1:
                st.markdown("---")
                col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
                
                with col1:
                    if page > 1:
                        if st.button("⏮️ First"):
                            st.session_state.current_page = 1
                            st.rerun()
                
                with col2:
                    if page > 1:
                        if st.button("◀️ Previous"):
                            st.session_state.current_page = page - 1
                            st.rerun()
                
                with col3:
                    new_page = st.number_input(
                        f"Page ({page} of {total_pages})", 
                        min_value=1, 
                        max_value=total_pages, 
                        value=page
                    )
                    if new_page != page:
                        st.session_state.current_page = new_page
                        st.rerun()
                
                with col4:
                    if page < total_pages:
                        if st.button("Next ▶️"):
                            st.session_state.current_page = page + 1
                            st.rerun()
                
                with col5:
                    if page < total_pages:
                        if st.button("Last ⏭️"):
                            st.session_state.current_page = total_pages
                            st.rerun()
        
        else:
            if search_term:
                st.info(f"No products found matching '{search_term}'")
            else:
                st.info("No products found in your store.")

# ---------------- TAB 4: Analytics ----------------
with tab4:
    st.header("📈 Advanced Analytics")
    
    if not (woo_url and woo_ck and woo_cs):
        st.warning("👉 Configure WooCommerce settings in the sidebar to view analytics.")
    else:
        # Date range selector
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30))
        with col2:
            end_date = st.date_input("End Date", value=datetime.now())
        
        if st.button("🔄 Refresh Analytics"):
            st.cache_data.clear()
            st.rerun()
        
        with st.spinner("Loading analytics data..."):
            # Fetch extended data for analytics
            products, total_products, _ = get_woo_products(woo_url, woo_ck, woo_cs, per_page=100)
            orders = get_woo_orders(woo_url, woo_ck, woo_cs, per_page=200)
            
            # Filter orders by date range
            filtered_orders = []
            for order in orders:
                order_date = order.get('date_created', '')[:10]
                if order_date:
                    try:
                        order_datetime = datetime.strptime(order_date, '%Y-%m-%d').date()
                        if start_date <= order_datetime <= end_date:
                            filtered_orders.append(order)
                    except:
                        continue
            
            metrics = calculate_metrics(filtered_orders, products)
        
        if filtered_orders or products:
            # Advanced metrics
            st.subheader("🔍 Detailed Analytics")
            
            # Product performance analysis
            if products:
                st.subheader("📊 Product Performance")
                
                # Create product performance DataFrame
                product_data = []
                for product in products:
                    # Count sales from orders
                    total_sold = 0
                    total_revenue = 0
                    
                    for order in filtered_orders:
                        for item in order.get('line_items', []):
                            if item.get('product_id') == product.get('id'):
                                total_sold += item.get('quantity', 0)
                                total_revenue += float(item.get('total', 0))
                    
                    product_data.append({
                        'Name': product.get('name', 'Unknown'),
                        'SKU': product.get('sku', 'N/A'),
                        'Price': float(product.get('regular_price', 0)),
                        'Stock': product.get('stock_quantity', 0),
                        'Units Sold': total_sold,
                        'Revenue': total_revenue,
                        'Status': product.get('stock_status', 'unknown')
                    })
                
                if product_data:
                    df_products = pd.DataFrame(product_data)
                    
                    # Top performing products chart
                    top_products = df_products.nlargest(10, 'Revenue')
                    
                    if not top_products.empty:
                        fig = px.bar(
                            top_products, 
                            x='Name', 
                            y='Revenue',
                            title="Top 10 Products by Revenue",
                            text='Units Sold'
                        )
                        fig.update_xaxes(tickangle=45)
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Product performance table
                    st.subheader("📋 Product Performance Table")
                    
                    # Add filters
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        status_filter = st.selectbox("Filter by Status", ["All", "instock", "outofstock", "onbackorder"])
                    with col2:
                        min_revenue = st.number_input("Min Revenue", min_value=0.0, value=0.0)
                    with col3:
                        sort_by = st.selectbox("Sort by", ["Revenue", "Units Sold", "Price", "Stock"])
                    
                    # Apply filters
                    filtered_df = df_products.copy()
                    if status_filter != "All":
                        filtered_df = filtered_df[filtered_df['Status'] == status_filter]
                    if min_revenue > 0:
                        filtered_df = filtered_df[filtered_df['Revenue'] >= min_revenue]
                    
                    # Sort
                    filtered_df = filtered_df.sort_values(sort_by, ascending=False)
                    
                    # Display table
                    st.dataframe(
                        filtered_df.style.format({
                            'Price': '${:.2f}',
                            'Revenue': '${:.2f}'
                        }), 
                        use_container_width=True
                    )
                    
                    # Export option
                    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Export Product Analytics",
                        csv_data,
                        file_name=f"product_analytics_{start_date}_to_{end_date}.csv",
                        mime="text/csv"
                    )
            
            # Sales trend analysis
            if filtered_orders:
                st.subheader("📈 Sales Trends")
                
                # Daily sales data
                daily_sales = {}
                current_date = start_date
                while current_date <= end_date:
                    daily_sales[current_date.strftime('%Y-%m-%d')] = {'orders': 0, 'revenue': 0.0}
                    current_date += timedelta(days=1)
                
                for order in filtered_orders:
                    order_date = order.get('date_created', '')[:10]
                    if order_date in daily_sales:
                        daily_sales[order_date]['orders'] += 1
                        daily_sales[order_date]['revenue'] += float(order.get('total', 0))
                
                # Create DataFrame for plotting
                trend_data = []
                for date, data in daily_sales.items():
                    trend_data.append({
                        'Date': date,
                        'Orders': data['orders'],
                        'Revenue': data['revenue']
                    })
                
                trend_df = pd.DataFrame(trend_data)
                trend_df['Date'] = pd.to_datetime(trend_df['Date'])
                
                # Dual axis chart
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=trend_df['Date'], 
                    y=trend_df['Revenue'],
                    name='Revenue',
                    line=dict(color='green'),
                    yaxis='y'
                ))
                
                fig.add_trace(go.Scatter(
                    x=trend_df['Date'], 
                    y=trend_df['Orders'],
                    name='Orders',
                    line=dict(color='blue'),
                    yaxis='y2'
                ))
                
                fig.update_layout(
                    title='Sales Trend Analysis',
                    xaxis_title='Date',
                    yaxis=dict(title='Revenue ($)', side='left'),
                    yaxis2=dict(title='Number of Orders', side='right', overlaying='y'),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Customer analysis
                st.subheader("👥 Customer Analysis")
                
                customer_data = {}
                for order in filtered_orders:
                    customer_email = order.get('billing', {}).get('email', 'Unknown')
                    if customer_email in customer_data:
                        customer_data[customer_email]['orders'] += 1
                        customer_data[customer_email]['total_spent'] += float(order.get('total', 0))
                    else:
                        customer_data[customer_email] = {
                            'orders': 1,
                            'total_spent': float(order.get('total', 0)),
                            'name': f"{order.get('billing', {}).get('first_name', '')} {order.get('billing', {}).get('last_name', '')}".strip() or 'Unknown'
                        }
                
                if customer_data:
                    # Top customers
                    top_customers = sorted(customer_data.items(), key=lambda x: x[1]['total_spent'], reverse=True)[:10]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**🏆 Top Customers by Revenue**")
                        for email, data in top_customers:
                            st.write(f"**{data['name']}** - ${data['total_spent']:.2f} ({data['orders']} orders)")
                    
                    with col2:
                        # Customer distribution
                        customer_df = pd.DataFrame([
                            {'Customer': data['name'], 'Orders': data['orders'], 'Total Spent': data['total_spent']}
                            for email, data in customer_data.items()
                        ])
                        
                        fig = px.histogram(
                            customer_df, 
                            x='Orders', 
                            title="Customer Order Distribution",
                            nbins=20
                        )
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("No data available for the selected date range.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🛒 <strong>WooCommerce Product Manager</strong> - Powered by AI</p>
    <p>Generate products • Manage inventory • Analyze performance • Grow your business</p>
</div>
""", unsafe_allow_html=True)
