import glob, os
import logging

from joblib import delayed, Parallel

from ..tools import misc

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def thin_line(line, odir):
    misc.run(f'lasthin -i {line} -step 1 -lowest -last_only -odir {odir} -odix _thin -olaz')


def thin_lines(idir, pattern, odir, n_jobs=None):
    os.makedirs(odir, exist_ok=True)
    logger.info(f'save thinned lines in : {odir}')
    if n_jobs is None:
        cpu_count = os.cpu_count()
        print(f"cpu_count {cpu_count}")
        n_jobs = max(1, int(cpu_count / 2))
    lines = glob.glob(os.path.join(idir, pattern))
    logger.info(f'Found {len(lines)} line(s) to proceed')
    Parallel(n_jobs=n_jobs, verbose=1)(delayed(thin_line)(line, odir) for line in lines)
