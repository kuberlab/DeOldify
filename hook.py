import numpy as np
import cv2


# from fastai.vision.data
# imagenet_stats
mean = np.array([0.485, 0.456, 0.406]).reshape([3, 1, 1])
std = np.array([0.229, 0.224, 0.225]).reshape([3, 1, 1])


class BaseFilter(object):
    def __init__(self, driver):
        super().__init__()
        self.driver = driver

    def denorm(self, x):
        return x * std + mean

    def norm(self, x):
        return (x-mean) / std

    def _transform(self, image):
        return image

    def _scale_to_square(self, orig, targ: int):
        # a simple stretch to fit a square really makes a big difference in rendering quality/consistency.
        # I've tried padding to the square as well (reflect, symetric, constant, etc).  Not as good!
        targ_sz = (targ, targ)
        return cv2.resize(orig, targ_sz, interpolation=cv2.INTER_AREA)

    def _get_model_ready_image(self, orig, sz: int):
        result = self._scale_to_square(orig, sz)
        result = self._transform(result)
        return result

    def _model_process(self, orig, sz: int):
        model_image = self._get_model_ready_image(orig, sz)
        x = np.divide(model_image.transpose([2, 0, 1]), 255)
        x = self.norm(x)
        x = np.expand_dims(x, axis=0)

        result = self.driver.predict({'0': x})

        out = result['0']
        out = self.denorm(out).clip(min=0, max=1)
        out = out.squeeze().transpose([1, 2, 0]) * 255
        out = out.astype(np.uint8)
        return out

    def _unsquare(self, image, orig):
        targ_sz = orig.shape[1], orig.shape[0]
        image = cv2.resize(image, targ_sz, interpolation=cv2.INTER_AREA)
        return image


class ColorizerFilter(BaseFilter):
    def __init__(self, driver, map_to_orig: bool = True):
        super().__init__(driver=driver)
        self.render_base = 16
        self.map_to_orig = map_to_orig

    def filter(self, orig_image, filtered_image, render_factor: int):
        render_sz = render_factor * self.render_base
        model_image = self._model_process(orig=filtered_image, sz=render_sz)

        if self.map_to_orig:
            return self._post_process(model_image, orig_image)
        else:
            return self._post_process(model_image, filtered_image)

    def _transform(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # This takes advantage of the fact that human eyes are much less sensitive to
    # imperfections in chrominance compared to luminance.  This means we can
    # save a lot on memory and processing in the model, yet get a great high
    # resolution result at the end.  This is primarily intended just for
    # inference
    def _post_process(self, raw_color, orig):
        raw_color = self._unsquare(raw_color, orig)
        color_np = raw_color
        orig_np = orig
        color_yuv = cv2.cvtColor(color_np, cv2.COLOR_BGR2YUV)
        # do a black and white transform first to get better luminance values
        if len(orig_np.shape) == 2:
            orig_np = np.expand_dims(orig_np, axis=2)

        orig_yuv = cv2.cvtColor(orig_np, cv2.COLOR_BGR2YUV)
        hires = np.copy(orig_yuv)
        hires[:, :, 1:3] = color_yuv[:, :, 1:3]
        final = cv2.cvtColor(hires, cv2.COLOR_YUV2BGR)
        return final


class MasterFilter(BaseFilter):
    def __init__(self, filters, render_factor: int):
        self.filters = filters
        self.render_factor = render_factor

    def filter(self, orig_image, filtered_image, render_factor: int = None):
        render_factor = self.render_factor if render_factor is None else render_factor

        for filter in self.filters:
            filtered_image = filter.filter(orig_image, filtered_image, render_factor)

        return filtered_image


render_factor = 35
filter: MasterFilter = None


PARAMS = {
    'render_factor': 35,
    'output_type': 'bytes',
}


def init_hook(**params):
    print(params)

    PARAMS.update(params)

    PARAMS['render_factor'] = int(PARAMS['render_factor'])


def _load_image(inputs, image_key):
    image = inputs.get(image_key)
    if image is None:
        raise RuntimeError('Missing "{0}" key in inputs. Provide an image in "{0}" key'.format(image_key))
    if len(image.shape) == 0:
        image = np.stack([image.tolist()])

    if len(image.shape) < 3:
        image = cv2.imdecode(np.frombuffer(image[0], np.uint8), cv2.IMREAD_COLOR)
        image = image[:, :, ::-1]

    return image


def process(inputs, ctx, **kwargs):
    image = _load_image(inputs, 'input')

    global filter
    if filter is None:
        filter = MasterFilter([ColorizerFilter(ctx.driver)], render_factor)

    filtered = filter.filter(np.copy(image), image, render_factor=render_factor)

    if PARAMS['output_type'] == 'bytes':
        filtered = filtered[:, :, ::-1]
        image_output = cv2.imencode(".jpg", filtered, params=[cv2.IMWRITE_JPEG_QUALITY, 95])[1].tostring()
    else:
        image_output = filtered

    return {'output': image_output}
