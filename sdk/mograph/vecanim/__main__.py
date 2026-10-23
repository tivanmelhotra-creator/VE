"""CLI:  python3 -m mograph.vecanim <command>

  make PANEL --shot NAME|FILE -o OUT [--preview] [--lottie] [--jobs N] [--rerig]
  rig  PANEL -o OUT            (trace + auto rig only; writes rig.json + rig_debug.png)
  templates                    (list shot templates)
"""
import sys, argparse
from . import pipeline as P


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == '_worker':
        P.worker(argv[1], int(argv[2]), int(argv[3]), argv[4]); return
    if argv and argv[0] == '_post':
        P.post(argv[1], argv[2]); return
    ap = argparse.ArgumentParser(prog='python3 -m mograph.vecanim', description='line art -> animated vector film')
    sub = ap.add_subparsers(dest='cmd', required=True)
    m = sub.add_parser('make'); m.add_argument('panel'); m.add_argument('--shot', default='duo_affection')
    m.add_argument('-o', '--out', default='/home/user/vec_out'); m.add_argument('--preview', action='store_true')
    m.add_argument('--lottie', action='store_true'); m.add_argument('--jobs', type=int); m.add_argument('--rerig', action='store_true')
    r = sub.add_parser('rig'); r.add_argument('panel'); r.add_argument('-o', '--out', default='/home/user/vec_out')
    r.add_argument('--rerig', action='store_true')
    sub.add_parser('templates')
    a = ap.parse_args(argv)
    if a.cmd == 'make':
        P.render(a.panel, a.shot, a.out, a.jobs, a.preview, a.rerig, a.lottie)
    elif a.cmd == 'rig':
        print(P.prepare(a.panel, a.out, a.rerig))
    else:
        print('\n'.join(P.templates()))


if __name__ == '__main__':
    main()
