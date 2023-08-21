from utils import config_parser
from load_dataset import load_blender



def run():
    '''
    Train the model
    '''
    parser = config_parser()
    args = parser.parse_args()

    # Load dataset
    images, poses, render_poses, hwf, i_split = load_blender(args.datadir, args.half_res, args.testskip)
    print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)

    
