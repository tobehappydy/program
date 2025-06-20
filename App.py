import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
from pathlib import Path
from dipy.io.image import load_nifti
from datetime import datetime
from dipy.io.gradients import read_bvals_bvecs
import nibabel as nib
# 功能函数导入
from brain_segmentation import brain_segmentation
from denoise_NLMEANS import denoise_NLMEANS
from enhance import enhance_histeq
from enhance_with_sharpening import sharpen_laplace
from mask_dilation import sharpen_laplace_dilation
from mask_erosion import sharpen_laplace_erosion
from visualize_mosaic import generate_slice_mosaic
from denoise_ASCM import denoise_ASCM
from smooth import smooth_gaussian
from wavelet_transform import wavelet_transform
from reconstruction_DTI import DTI_reconstruction
from tissue_segmentation import tissue_segmentation

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'result'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB limit

# Ensure upload and result directories exist
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
Path(app.config['RESULT_FOLDER']).mkdir(exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        return jsonify({
            'message': 'File uploaded successfully',
            'filepath': filepath,
            'filename': filename
        })

    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/segment', methods=['POST'])
def process_segmentation():
    data = request.get_json()
    filepath = data.get('filepath')

    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'Invalid file path'}), 400

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(app.config['RESULT_FOLDER']) / timestamp
        output_dir.mkdir(exist_ok=True)

        data, affine = load_nifti(filepath)
        data = np.squeeze(data)

        output_files = brain_segmentation(data, affine, output_dir)

        preview_path = output_files.get('segmented_preview')
        if preview_path and os.path.exists(preview_path):
            preview_url = f"/results/{timestamp}/{os.path.basename(preview_path)}"
        else:
            preview_url = None

        return jsonify({
            'message': 'Segmentation completed successfully',
            'result_dir': timestamp,
            'preview_url': preview_url,
            'output_files': output_files
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/denoise', methods=['POST'])
def process_denoise():
    data = request.get_json()
    filepath = data.get('filepath')

    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'Invalid file path'}), 400

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(app.config['RESULT_FOLDER']) / timestamp
        output_dir.mkdir(exist_ok=True)

        data, affine = load_nifti(filepath)
        data = np.squeeze(data)
        mask = np.ones(data.shape, dtype=bool)

        output_files = denoise_NLMEANS(data, mask, affine, output_dir)

        preview_path = output_files.get('denoised_preview')
        if preview_path and os.path.exists(preview_path):
            preview_url = f"/results/{timestamp}/{os.path.basename(preview_path)}"
        else:
            preview_url = None

        return jsonify({
            'message': 'Denoising completed successfully',
            'result_dir': timestamp,
            'preview_url': preview_url,
            'output_files': output_files
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/enhance', methods=['POST'])
def process_enhance():
    data = request.get_json()
    filepath = data.get('filepath')

    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'Invalid file path'}), 400

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(app.config['RESULT_FOLDER']) / timestamp
        output_dir.mkdir(exist_ok=True)

        data, affine = load_nifti(filepath)
        data = np.squeeze(data)

        output_files = enhance_histeq(data, affine, output_dir, num_bins=256)

        preview_path = output_files.get('enhanced_preview')
        if preview_path and os.path.exists(preview_path):
            preview_url = f"/results/{timestamp}/{os.path.basename(preview_path)}"
        else:
            preview_url = None

        return jsonify({
            'message': 'Enhancement completed successfully',
            'result_dir': timestamp,
            'preview_url': preview_url,
            'output_files': output_files
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/enhance_with_sharpening', methods=['POST'])
def process_enhance_with_sharpening():
    data = request.get_json()
    filepath = data.get('filepath')

    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'Invalid file path'}), 400

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(app.config['RESULT_FOLDER']) / timestamp
        output_dir.mkdir(exist_ok=True)

        data, affine = load_nifti(filepath)
        data = np.squeeze(data)
        mask = np.ones(data.shape, dtype=bool)

        output_files = sharpen_laplace(data, mask, affine, output_dir, sharpening_factor=0.5, sigma=1.0)

        preview_path = output_files.get('sharpened_preview')
        if preview_path and os.path.exists(preview_path):
            preview_url = f"/results/{timestamp}/{os.path.basename(preview_path)}"
        else:
            preview_url = None

        return jsonify({
            'message': 'Sharpening enhancement completed successfully',
            'result_dir': timestamp,
            'preview_url': preview_url,
            'output_files': output_files
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/erosion', methods=['POST'])
def process_erosion():

    data = request.get_json()
    filepath = data.get('filepath')
    sharpening_factor = data.get('sharpening_factor', 0.5)
    sigma = data.get('sigma', 1.0)

    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'Invalid file path'}), 400

    try:
        # 创建结果目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(app.config['RESULT_FOLDER']) / timestamp
        output_dir.mkdir(exist_ok=True)

        # 读取原始图像
        img_data, affine = load_nifti(filepath)
        img_data = np.squeeze(img_data)

        # 用阈值生成掩膜，非零像素都当脑组织
        mask = img_data > 0

        # 然后再传给腐蚀+锐化函数
        output_files = sharpen_laplace_erosion(
            img_data, mask, affine, output_dir,
            sharpening_factor=sharpening_factor,
            sigma=sigma
        )

        preview_path = output_files.get('comparison_eroded_preview')
        if preview_path and os.path.exists(preview_path):
            preview_url = f"/results/{timestamp}/{os.path.basename(preview_path)}"
        else:
            preview_url = None

        return jsonify({
            'message': 'Erosion with sharpening completed successfully',
            'result_dir': timestamp,
            'preview_url': preview_url,
            'output_files': output_files
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/dilation', methods=['POST'])
def process_dilation():
    data = request.get_json()
    filepath = data.get('filepath')
    sharpening_factor = data.get('sharpening_factor', 0.5)
    sigma = data.get('sigma', 1.0)

    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'Invalid file path'}), 400

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(app.config['RESULT_FOLDER']) / timestamp
        output_dir.mkdir(exist_ok=True)

        img_data, affine = load_nifti(filepath)
        img_data = np.squeeze(img_data)

        # 用阈值生成掩膜，非零像素都当脑组织
        mask = img_data > 0

        output_files = sharpen_laplace_dilation(
            img_data, mask, affine, output_dir,
            sharpening_factor=sharpening_factor,
            sigma=sigma
        )

        # 生成预览图 URL
        preview_path = output_files.get('comparison_dilated_preview')
        if preview_path and os.path.exists(preview_path):
            preview_url = f"/results/{timestamp}/{os.path.basename(preview_path)}"
        else:
            preview_url = None


        # 返回结果
        return jsonify({
            'message': 'Dilation with sharpening completed successfully',
            'result_dir': timestamp,
            'preview_url': preview_url,
            'output_files': output_files
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/denoise_ASCM', methods=['POST'])
def process_denoise_ASCM():
    data = request.get_json()
    filepath = data.get('filepath')

    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'Invalid file path'}), 400

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(app.config['RESULT_FOLDER']) / timestamp
        output_dir.mkdir(exist_ok=True)

        data, affine = load_nifti(filepath)
        data = np.squeeze(data)
        mask = np.ones(data.shape, dtype=bool)
        output_files = denoise_ASCM(data, mask, affine, output_dir)

        preview_path = output_files.get('denoised_ascm_preview')
        if preview_path and os.path.exists(preview_path):
            preview_url = f"/results/{timestamp}/{os.path.basename(preview_path)}"
        else:
            preview_url = None

        return jsonify({
            'message': 'ASCM Denoising completed successfully',
            'result_dir': timestamp,
            'preview_url': preview_url,
            'output_files': output_files
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/gaussian_smooth', methods=['POST'])
def process_gaussian():
    data = request.get_json()
    filepath = data.get('filepath')
    sigma = float(data.get('sigma', 1.0))

    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'Invalid file path'}), 400

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(app.config['RESULT_FOLDER']) / timestamp
        output_dir.mkdir(exist_ok=True)

        data, affine = load_nifti(filepath)
        data = np.squeeze(data)

        output_files = smooth_gaussian(data, affine, output_dir, sigma=sigma)


        preview_path = output_files.get('smoothed_preview')
        if preview_path and os.path.exists(preview_path):
            preview_url = f"/results/{timestamp}/{os.path.basename(preview_path)}"
        else:
            preview_url = None

        return jsonify({
            'message': 'Gaussian smoothing completed successfully',
            'result_dir': timestamp,
            'preview_url': preview_url,
            'output_files': output_files
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/wavelet_transform', methods=['POST'])
def process_wavelet():
    data = request.get_json()
    filepath = data.get('filepath')
    wavelet = data.get('wavelet', 'haar')
    level = int(data.get('level', 1))

    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'Invalid file path'}), 400

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(app.config['RESULT_FOLDER']) / timestamp
        output_dir.mkdir(exist_ok=True)

        data, affine = load_nifti(filepath)
        data = np.squeeze(data)

        output_files = wavelet_transform(data, affine, output_dir, wavelet=wavelet, level=level)

        preview_path = output_files.get('wavelet_preview')
        if preview_path and os.path.exists(preview_path):
            preview_url = f"/results/{timestamp}/{os.path.basename(preview_path)}"
        else:
            preview_url = None

        return jsonify({
            'message': 'Wavelet transform completed successfully',
            'result_dir': timestamp,
            'preview_url': preview_url,
            'output_files': output_files
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/tissue_segmentation', methods=['POST'])
def process_tissue_segmentation():
    data = request.get_json()
    filepath = data.get('filepath')
    nclass = int(data.get('nclass', 3))  # Default to 3 tissue classes
    beta = float(data.get('beta', 0.1))  # Default beta value

    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'Invalid file path'}), 400

    try:
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(app.config['RESULT_FOLDER']) / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load the NIfTI file
        img = nib.load(filepath)
        data = img.get_fdata()
        affine = img.affine

        # Perform tissue segmentation
        output_files = tissue_segmentation(data, affine, output_dir, nclass=nclass, beta=beta)

        # Generate URLs for the preview image
        preview_path = output_files.get('tissue_segmentation_preview')
        if preview_path and os.path.exists(preview_path):
            preview_url = f"/results/{timestamp}/{os.path.basename(preview_path)}"
        else:
            preview_url = None

        return jsonify({
            'message': 'Tissue segmentation completed successfully',
            'result_dir': timestamp,
            'preview_url': preview_url,
            'output_files': {
                'segmentation_result': f"/results/{timestamp}/tissue_segmentation_processed.nii.gz",
                'gm_probability': f"/results/{timestamp}/gm_probability.nii.gz",
                'wm_probability': f"/results/{timestamp}/wm_probability.nii.gz",
                'preview_image': preview_url
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/dti_reconstruct', methods=['POST'])
def dti_reconstruct():
    # 校验输入
    dwi_filepath = request.form.get('dwi_filepath', '').strip()
    bvec_file = request.files.get('bvec_file')
    bval_file = request.files.get('bval_file')

    if not dwi_filepath or not os.path.exists(dwi_filepath):
        return jsonify({'error': 'Invalid or missing DWI file path'}), 400
    if bvec_file is None or bval_file is None:
        return jsonify({'error': 'Both .bvec and .bval files are required'}), 400

    try:

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(app.config['RESULT_FOLDER']) / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存 bvec / bval
        bvec_fname = output_dir / secure_filename(bvec_file.filename or 'dwi.bvec')
        bval_fname = output_dir / secure_filename(bval_file.filename or 'dwi.bval')
        bvec_file.save(bvec_fname)
        bval_file.save(bval_fname)

        dwi_data, affine = load_nifti(dwi_filepath)
        dwi_data = np.squeeze(dwi_data)

        # 读取梯度表
        bvals, bvecs = read_bvals_bvecs(str(bval_fname), str(bvec_fname))

        # 调用 DTI 重建
        output_files = DTI_reconstruction(
            dwi_data=dwi_data,
            bvals=bvals,
            bvecs=bvecs,
            affine=affine,
            output_dir=output_dir,
            # 下面两个参数可按需改动
            median_radius=3,
            numpass=1,
            autocrop=True,
            dilate=2
        )

        preview_path = output_files.get('dti_preview')
        preview_url = (
            f"/results/{timestamp}/{Path(preview_path).name}"
            if preview_path and os.path.exists(preview_path)
            else None
        )

        app.logger.warning(f"[DTI] preview_url SENT → {preview_url}")
        return jsonify({
            'message': 'DTI reconstruction completed successfully',
            'result_dir': timestamp,
            'preview_url': preview_url,

            **output_files

        })
    except Exception as e:
        app.logger.exception('DTI reconstruction failed')
        return jsonify({'error': str(e)}), 500

@app.route('/view_slices', methods=['POST'])
def view_slices():
    data = request.get_json()
    filepath = data.get('filepath')
    result_dir = data.get('result_dir')

    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'Invalid file path'}), 400

    try:
        img_data, affine = load_nifti(filepath)
        img_data = np.squeeze(img_data)

        output_dir = Path(app.config['RESULT_FOLDER']) / result_dir
        output_dir.mkdir(exist_ok=True)

        mosaic_path = output_dir / "mosaic.png"
        generate_slice_mosaic(img_data, mosaic_path, rows=6, cols=10, spacing=2)

        mosaic_url = f"/results/{result_dir}/mosaic.png"

        return jsonify({
            'message': 'Mosaic generated',
            'mosaic_url': mosaic_url
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/download_results', methods=['GET'])
def download_results():
    result_dir = request.args.get('result_dir')

    if not result_dir:
        return jsonify({'error': 'Missing result directory parameter'}), 400

    result_path = Path(app.config['RESULT_FOLDER']) / result_dir
    if not result_path.exists():
        return jsonify({'error': 'Result directory not found'}), 404

    # 只下载第一个找到的文件
    files = list(result_path.glob('*'))
    if not files:
        return jsonify({'error': 'No files to download'}), 404

    return send_from_directory(str(result_path), files[0].name, as_attachment=True)


def allowed_file(filename):
    return '.' in filename and filename.lower().endswith(('.nii', '.nii.gz'))


@app.route('/results/<path:filename>')
def serve_result(filename):
    result_dir, *rest = filename.split('/')
    filepath = '/'.join(rest)

    full_dir = Path(app.config['RESULT_FOLDER']) / result_dir
    if not full_dir.exists():
        return jsonify({'error': 'Result directory not found'}), 404

    return send_from_directory(str(full_dir), filepath)


def allowed_file(filename):
    return '.' in filename and filename.lower().endswith(('.nii', '.nii.gz'))


if __name__ == '__main__':
    app.run(debug=True)