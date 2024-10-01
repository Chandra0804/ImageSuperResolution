from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

app = Flask(__name__)
CORS(app)

class ChannelAttention(nn.Module):
    def __init__(self, num_features, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_features, num_features // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_features // reduction, num_features, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class RCAB(nn.Module):
    def __init__(self, num_features, reduction):
        super(RCAB, self).__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.ca = ChannelAttention(num_features, reduction)

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        res = self.ca(res)
        return res + x

class RRDB(nn.Module):
    def __init__(self, num_features, num_grow_channels, reduction):
        super(RRDB, self).__init__()
        self.rcab1 = RCAB(num_features, reduction)
        self.rcab2 = RCAB(num_features, reduction)
        self.rcab3 = RCAB(num_features, reduction)
        self.conv = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.rcab1(x)
        out = self.rcab2(out)
        out = self.rcab3(out)
        out = self.conv(out)
        return out * 0.2 + x

class EASRNGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, num_features, num_blocks, reduction=16, scale_factor=4):
        super(EASRNGenerator, self).__init__()
        self.conv_first = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[RRDB(num_features, num_features // 2, reduction) for _ in range(num_blocks)])
        self.conv_body = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        
        # Upsampling layers
        self.upconv1 = nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1)
        self.upconv2 = nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        self.conv_hr = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.conv_last = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.body(feat)
        body_feat = self.conv_body(body_feat)
        feat = feat + body_feat
        
        feat = self.lrelu(self.pixel_shuffle(self.upconv1(feat)))
        feat = self.lrelu(self.pixel_shuffle(self.upconv2(feat)))
        
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EASRNGenerator(3, 3, 64, 23, reduction=16, scale_factor=4).to(device)
model.load_state_dict(torch.load('/Users/saiteja/project/ImageSuperResolution/easrn_generator_epoch_300.pth', map_location=device))
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.ToTensor(),
])

@app.route('/super-resolve', methods=['POST'])
def super_resolve():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    
    try:
        image = Image.open(io.BytesIO(file.read()))
    except IOError:
        return jsonify({'error': 'Invalid image file'}), 400
    
    # Convert to RGB if the image is not in RGB mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transformation
    image_tensor = transform(image).unsqueeze(0).to(device)

    try:
        with torch.no_grad():
            output = model(image_tensor)
            output = output.squeeze(0).clamp(0, 1)
    except RuntimeError as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

    # Convert output to PIL Image
    output_image = transforms.ToPILImage()(output.cpu())
    output_bytes = io.BytesIO()
    output_image.save(output_bytes, format='PNG')
    output_bytes.seek(0)

    return send_file(output_bytes, mimetype='image/png', as_attachment=True, download_name='super_resolved.png')

@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)