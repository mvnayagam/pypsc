import logging

logger = logging.getLogger(__name__)


class ChoiceOfOrigin:
    """
    Choice of Origin (COO) correction.

    If first order reflection has negative signed amplitude,
    origin is shifted by (0.5,0.5,...,0.5).

    Effect:
        odd l  -> amplitude sign inversion
        even l -> unchanged
    """

    def __init__(self):
        self.shift_applied = False

    def _get_l_key(self, reflection_map):
        
        for reflection in reflection_map:
            
            if self._get_l(reflection) == 1:
                return reflection
            
        return None
    
    def _get_l(self, reflection):
        
        if isinstance(reflection, tuple):
            return reflection[-1]
        
        return reflection

    def apply(self, amplitudes_reflection_map):
        
        corrected = amplitudes_reflection_map.copy()
        
        l1 = self._get_l_key(amplitudes_reflection_map)
        
        if l1 is None:
            logger.warning("First order reflection l=1 not found. COO skipped")
            return corrected
        
        if amplitudes_reflection_map[l1] < 0:
            
            self.shift_applied = True
            
            logger.info(f"COO applied using reflection {l1} : g(l) = {amplitudes_reflection_map[l1]:.4f}" )
            
            for reflection, value in amplitudes_reflection_map.items():
                l = self._get_l(reflection)
                if l % 2 == 1:
                    corrected[reflection] = -value
        else:

            logger.info( f"COO not required g(l={l1})={amplitudes_reflection_map[l1]:.4f}")
        
        return corrected
    