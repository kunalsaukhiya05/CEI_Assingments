import markdown
import re
import base64
import os

# Install markdown if not present
try:
    import markdown
except ImportError:
    os.system('pip install markdown')
    import markdown

def get_base64_image(image_path):
    if not os.path.exists(image_path):
        return ""
    
    with open(image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
    
    ext = os.path.splitext(image_path)[1][1:].lower()
    if ext == 'jpg':
        ext = 'jpeg'
    return f"data:image/{ext};base64,{encoded_string}"

def markdown_to_html_with_base64_images(md_file, output_html):
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Find all markdown images: ![alt](path)
    # Using a simple regex
    pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
    
    def replacer(match):
        alt_text = match.group(1)
        image_path = match.group(2)
        base64_data = get_base64_image(image_path)
        if base64_data:
            return f'<img src="{base64_data}" alt="{alt_text}" style="max-width:100%;">'
        return match.group(0) # Return original if not found
    
    md_content_replaced = re.sub(pattern, replacer, md_content)
    
    html_body = markdown.markdown(md_content_replaced)
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Final Project Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; padding: 40px; max-width: 900px; margin: 0 auto; color: #333; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            h1 {{ border-bottom: 2px solid #2c3e50; padding-bottom: 10px; }}
            h2 {{ border-bottom: 1px solid #ccc; padding-bottom: 5px; margin-top: 30px; }}
            img {{ display: block; margin: 20px auto; border: 1px solid #ddd; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }}
            code {{ background-color: #f4f4f4; padding: 2px 5px; border-radius: 3px; font-family: monospace; }}
            blockquote {{ border-left: 4px solid #ccc; margin: 0; padding-left: 15px; color: #666; font-style: italic; }}
        </style>
    </head>
    <body>
        {html_body}
    </body>
    </html>
    """
    
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Successfully generated {output_html}")

if __name__ == "__main__":
    markdown_to_html_with_base64_images("Report.md", "Final_Report.html")
