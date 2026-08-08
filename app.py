# import os
# import torch
# from flask import Flask, render_template, request, redirect, url_for, send_from_directory
# from flask_wtf import FlaskForm
# from flask_bootstrap import Bootstrap
# from werkzeug.utils import secure_filename
# from wtforms import FileField, SubmitField, FloatField, HiddenField
# from wtforms.validators import InputRequired
# from PIL import Image
# from torchvision import transforms
# import io

# # Import your existing AdaIN code
# from utils.models import VGGEncoder, Decoder
# from utils.utils import adaptive_instance_normalization, calc_mean_std


# app = Flask(__name__)
# app.config['SECRET_KEY'] = 'supersecretkey'
# app.config['UPLOAD_FOLDER'] = 'static/uploads'
# app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
# Bootstrap(app)

# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# class UploadForm(FlaskForm):
#     content = FileField('Content Image')
#     style = FileField('Style Image')
#     content_path = HiddenField()
#     style_path = HiddenField()
#     alpha = FloatField('Alpha', default=1.0)
#     submit = SubmitField('Transfer Style')

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# encoder = VGGEncoder('vgg_normalised.pth').to(device)

# decoder = Decoder().to(device)

# # decoder.load_state_dict(torch.load('experiment/trial1/decoder_6.pth'))
# decoder.load_state_dict(
#     torch.load(
#         "experiment/trial1/decoder_6.pth",
#         map_location=device
#     )
# )
# encoder.eval()
# decoder.eval()

# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# def style_transfer(content_image, style_image, encoder, decoder, alpha, device):
#     # content_transform = transforms.Compose([
#     #     transforms.Resize(512),
#     #     transforms.ToTensor()
#     # ])

#     # style_transform = transforms.Compose([
#     #     transforms.Resize(512),
#     #     transforms.ToTensor()
#     # ])
    
#     content_transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor()
#     ])

#     style_transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor()
#     ])
#     content_image = content_transform(content_image).unsqueeze(0).to(device)
#     style_image = style_transform(style_image).unsqueeze(0).to(device)

#     with torch.no_grad():
#         content_feats = encoder(content_image, is_test=True)
#         style_feats = encoder(style_image, is_test=True)

#         stylized_feats = adaptive_instance_normalization(content_feats, style_feats)

#         stylized_feats = alpha * stylized_feats + (1 - alpha) * content_feats

#         stylized_image = decoder(stylized_feats)

#     return stylized_image


# def save_image(image, path):
#     image = image.cpu().clone()
#     image = image.squeeze(0)
#     image = image.clamp(0, 1)
#     image = transforms.ToPILImage()(image)
#     image.save(path)



# @app.route('/', methods=['GET', 'POST'])
# def index():
#     form = UploadForm()
#     result_image = None
#     content_filename = None
#     style_filename = None
#     error = None

#     if form.validate_on_submit():
#         if form.content.data and form.content.data.filename:
#             if allowed_file(form.content.data.filename):
#                 content_filename = secure_filename(form.content.data.filename)
#                 form.content.data.save(os.path.join(app.config['UPLOAD_FOLDER'], content_filename))
#                 form.content_path.data = content_filename
#         else:
#             content_filename = form.content_path.data

#         if form.style.data and form.style.data.filename:
#             if allowed_file(form.style.data.filename):
#                 style_filename = secure_filename(form.style.data.filename)
#                 form.style.data.save(os.path.join(app.config['UPLOAD_FOLDER'], style_filename))
#                 form.style_path.data = style_filename
#         else:
#             style_filename = form.style_path.data

#         if content_filename and style_filename:
#             content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
#             style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)
            
#             try:
#                 content_image = Image.open(content_path).convert('RGB')
#                 style_image = Image.open(style_path).convert('RGB')

#                 alpha = float(form.alpha.data)
#                 stylized_image = style_transfer(content_image, style_image, encoder, decoder, alpha, device)

#                 result_filename = 'stylized_' + content_filename
#                 result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
#                 save_image(stylized_image, result_path)
                
#                 result_image = result_filename
#             except Exception as e:
#                 error = str(e)
#     else:
#         if not content_filename:
#             error = 'Please upload content image'
#         if not style_filename:
#             error = 'Please upload style image'

#     return render_template('index.html', form=form, result_image=result_image, content_image=content_filename,
#                            style_image=style_filename, error=error)


# @app.route('/uploads/<filename>')
# def send_image(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# @app.route('/examples/<path:filename>')
# def send_example(filename):
#     return send_from_directory('examples', filename)


# if __name__ == '__main__':
#     from werkzeug.serving import run_simple
#     run_simple('localhost', 5000, app, use_reloader=True, use_debugger=True)





import os
import gc

import torch
from flask import (
    Flask,
    render_template,
    send_from_directory,
)
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename

from wtforms import FileField, SubmitField, FloatField, HiddenField
from PIL import Image
from torchvision import transforms

from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization


# ============================================================
# FLASK CONFIGURATION
# ============================================================

app = Flask(__name__)

app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY",
    "supersecretkey"
)

app.config["UPLOAD_FOLDER"] = "static/uploads"

app.config["ALLOWED_EXTENSIONS"] = {
    "png",
    "jpg",
    "jpeg",
}

Bootstrap(app)

os.makedirs(
    app.config["UPLOAD_FOLDER"],
    exist_ok=True
)


# ============================================================
# FORM
# ============================================================

class UploadForm(FlaskForm):

    content = FileField("Content Image")

    style = FileField("Style Image")

    content_path = HiddenField()

    style_path = HiddenField()

    alpha = FloatField(
        "Alpha",
        default=1.0
    )

    submit = SubmitField(
        "Transfer Style"
    )


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print(f"Device: {DEVICE}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("=" * 60)


# ============================================================
# MODEL PATHS
# ============================================================

VGG_PATH = "vgg_normalised.pth"

DECODER_PATH = "experiment/trial1/decoder_6.pth"


# ============================================================
# LOAD MODELS ONCE
# ============================================================

print("Loading VGG encoder...")

encoder = VGGEncoder(
    VGG_PATH
).to(DEVICE)

print("Loading decoder...")

decoder = Decoder().to(DEVICE)

print("Loading decoder checkpoint...")

decoder.load_state_dict(
    torch.load(
        DECODER_PATH,
        map_location=DEVICE
    )
)

# Evaluation mode
encoder.eval()
decoder.eval()

# Make sure parameters don't require gradients.
for parameter in encoder.parameters():
    parameter.requires_grad_(False)

for parameter in decoder.parameters():
    parameter.requires_grad_(False)

print("Models loaded successfully.")


# ============================================================
# IMAGE TRANSFORMS
# ============================================================

# IMPORTANT:
# 256x256 uses considerably less memory than 512x512.
#
# Render Free has limited RAM, so start with 256x256.

content_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

style_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(
            ".",
            1
        )[1].lower()
        in app.config["ALLOWED_EXTENSIONS"]
    )


def save_image(image, path):

    image = image.detach().cpu()

    image = image.squeeze(0)

    image = image.clamp(0, 1)

    image = transforms.ToPILImage()(image)

    image.save(path)


# ============================================================
# ADAIN STYLE TRANSFER
# ============================================================

def style_transfer(
    content_image,
    style_image,
    alpha
):

    # --------------------------------------------------------
    # Prepare images
    # --------------------------------------------------------

    content_image = content_image.convert("RGB")

    style_image = style_image.convert("RGB")

    content_tensor = content_transform(
        content_image
    ).unsqueeze(0)

    style_tensor = style_transform(
        style_image
    ).unsqueeze(0)

    # Move tensors to device
    content_tensor = content_tensor.to(
        DEVICE,
        non_blocking=True
    )

    style_tensor = style_tensor.to(
        DEVICE,
        non_blocking=True
    )

    # --------------------------------------------------------
    # Clamp alpha
    # --------------------------------------------------------

    alpha = max(
        0.0,
        min(
            1.0,
            float(alpha)
        )
    )

    try:

        # ----------------------------------------------------
        # INFERENCE
        # ----------------------------------------------------

        with torch.inference_mode():

            # Content features
            content_feats = encoder(
                content_tensor,
                is_test=True
            )

            # Style features
            style_feats = encoder(
                style_tensor,
                is_test=True
            )

            # AdaIN
            stylized_feats = (
                adaptive_instance_normalization(
                    content_feats,
                    style_feats
                )
            )

            # Alpha blending
            stylized_feats = (
                alpha * stylized_feats
                +
                (1.0 - alpha) * content_feats
            )

            # Decoder
            stylized_image = decoder(
                stylized_feats
            )

            # ------------------------------------------------
            # MOVE OUTPUT TO CPU BEFORE CLEANUP
            # ------------------------------------------------

            result = (
                stylized_image
                .detach()
                .cpu()
                .squeeze(0)
                .clamp(0, 1)
            )

        # ----------------------------------------------------
        # Convert Tensor -> PIL
        # ----------------------------------------------------

        result = transforms.ToPILImage()(result)

        return result

    finally:

        # ----------------------------------------------------
        # FREE MEMORY
        # ----------------------------------------------------

        for variable_name in [
            "content_tensor",
            "style_tensor",
            "content_feats",
            "style_feats",
            "stylized_feats",
            "stylized_image",
        ]:

            if variable_name in locals():

                del locals()[variable_name]

        gc.collect()

        # CUDA cleanup only if GPU exists
        if torch.cuda.is_available():

            torch.cuda.empty_cache()


# ============================================================
# HOME ROUTE
# ============================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def index():

    form = UploadForm()

    result_image = None

    content_filename = None

    style_filename = None

    error = None

    # --------------------------------------------------------
    # FORM SUBMISSION
    # --------------------------------------------------------

    if form.validate_on_submit():

        # ====================================================
        # CONTENT IMAGE
        # ====================================================

        if (
            form.content.data
            and form.content.data.filename
        ):

            if allowed_file(
                form.content.data.filename
            ):

                content_filename = secure_filename(
                    form.content.data.filename
                )

                content_path = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    content_filename
                )

                form.content.data.save(
                    content_path
                )

                form.content_path.data = (
                    content_filename
                )

            else:

                error = (
                    "Invalid content image format."
                )

        else:

            content_filename = (
                form.content_path.data
            )

        # ====================================================
        # STYLE IMAGE
        # ====================================================

        if (
            form.style.data
            and form.style.data.filename
        ):

            if allowed_file(
                form.style.data.filename
            ):

                style_filename = secure_filename(
                    form.style.data.filename
                )

                style_path = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    style_filename
                )

                form.style.data.save(
                    style_path
                )

                form.style_path.data = (
                    style_filename
                )

            else:

                error = (
                    "Invalid style image format."
                )

        else:

            style_filename = (
                form.style_path.data
            )

        # ====================================================
        # RUN STYLE TRANSFER
        # ====================================================

        if (
            content_filename
            and style_filename
            and error is None
        ):

            content_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                content_filename
            )

            style_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                style_filename
            )

            try:

                # Open images
                content_image = Image.open(
                    content_path
                ).convert("RGB")

                style_image = Image.open(
                    style_path
                ).convert("RGB")

                # Alpha
                alpha = float(
                    form.alpha.data
                )

                # ------------------------------------------------
                # ADAIN
                # ------------------------------------------------

                stylized_image = style_transfer(
                    content_image,
                    style_image,
                    alpha,
                   
                )

                # ------------------------------------------------
                # SAVE RESULT
                # ------------------------------------------------

                result_filename = (
                    "stylized_"
                    + content_filename
                )

                result_path = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    result_filename
                )

                stylized_image.save(
                    result_path
                )

                result_image = (
                    result_filename
                )

                # Close PIL images
                content_image.close()
                style_image.close()

                del stylized_image

                gc.collect()

            except Exception as e:

                print(
                    f"Style transfer error: {e}"
                )

                error = str(e)

    # ========================================================
    # RENDER TEMPLATE
    # ========================================================

    return render_template(
        "index.html",
        form=form,
        result_image=result_image,
        content_image=content_filename,
        style_image=style_filename,
        error=error
    )


# ============================================================
# UPLOADS
# ============================================================

@app.route(
    "/uploads/<filename>"
)
def send_image(filename):

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


# ============================================================
# EXAMPLES
# ============================================================

@app.route(
    "/examples/<path:filename>"
)
def send_example(filename):

    return send_from_directory(
        "examples",
        filename
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return {
        "status": "ok",
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available()
    }


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                5000
            )
        ),
        debug=False
    )



