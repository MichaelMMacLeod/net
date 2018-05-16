(setf *random-state* (make-random-state t))

(defun last1 (lst) 
  (first (last lst)))

(defun iterate (dimensions function)
  (let ((result (make-array dimensions)))
    (loop for row below (first dimensions)
          do (loop for column below (second dimensions)
                   do (funcall function result row column)))
    result))

(defun matrix-map (function matrix)
  (iterate 
    (array-dimensions matrix)
    (lambda (result row column)
      (setf (aref result row column)
            (funcall function (aref matrix row column))))))

(defun print-matrix (matrix &optional label)
  (if label (format t "~%~a" label))
  (terpri)
  (let ((dimensions (array-dimensions matrix)))
    (loop for row below (first dimensions)
          do (loop for column below (second dimensions)
                   do (format t "~,4@f " (aref matrix row column)))
          (terpri))))

(defun print-matricies (matricies &optional label)
  (if label (format t "~%~a" label))
  (terpri)
  (loop for matrix in matricies
        do (print-matrix matrix)))

(defun add (matrix-a matrix-b)
  (let ((dimensions-a (array-dimensions matrix-a))
        (dimensions-b (array-dimensions matrix-b)))
    (assert (equal dimensions-a dimensions-b))
    (iterate
      dimensions-a
      (lambda (result row column)
        (setf (aref result row column)
              (+ (aref matrix-a row column)
                 (aref matrix-b row column)))))))

(defun add-column (matrix-a matrix-b)
  (let ((dimensions-a (array-dimensions matrix-a))
        (dimensions-b (array-dimensions matrix-b)))
    (assert (and (= 1 (second dimensions-a))
                 (= (first dimensions-a) (first dimensions-b))))
    (iterate
      dimensions-b
      (lambda (result row column)
        (setf (aref result row column)
              (+ (aref matrix-a row 0)
                 (aref matrix-b row column)))))))

(defun average-columns (matrix)
  (iterate
    (list (first (array-dimensions matrix)) 1)
    (lambda (result row column)
      (incf (aref result row 0)
            (/ (aref matrix row column)
               (second (array-dimensions matrix)))))))

(defun subtract (matrix-a matrix-b)
  (let ((dimensions-a (array-dimensions matrix-a))
        (dimensions-b (array-dimensions matrix-b)))
    (assert (equal dimensions-a dimensions-b))
    (iterate
      dimensions-a
      (lambda (result row column)
        (setf (aref result row column)
              (- (aref matrix-a row column)
                 (aref matrix-b row column)))))))

(defun multiply (matrix-a matrix-b)
  (let ((dimensions-a (array-dimensions matrix-a))
        (dimensions-b (array-dimensions matrix-b)))
    (assert (= (second dimensions-a) (first dimensions-b)))
    (let ((result-dimensions (list (first dimensions-a)
                                   (second dimensions-b))))
      (iterate
        result-dimensions
        (lambda (result row column)
          (loop for i below (second dimensions-a)
                do (incf (aref result row column)
                         (* (aref matrix-a row i)
                            (aref matrix-b i column)))))))))

(defun hadamard (matrix-a matrix-b)
  (let ((dimensions-a (array-dimensions matrix-a))
        (dimensions-b (array-dimensions matrix-b)))
    (assert (equal dimensions-a dimensions-b))
    (iterate
      dimensions-a
      (lambda (result row column)
        (setf (aref result row column)
              (* (aref matrix-a row column)
                 (aref matrix-b row column)))))))

(defun square (matrix)
  (iterate
    (array-dimensions matrix)
    (lambda (result row column)
      (setf (aref result row column)
            (* (aref matrix row column)
               (aref matrix row column))))))

(defun transpose (matrix)
  (iterate
    (reverse (array-dimensions matrix))
    (lambda (result row column)
      (setf (aref result row column)
            (aref matrix column row)))))

(defun scale (matrix scalar)
  (iterate
    (array-dimensions matrix)
    (lambda (result row column)
      (setf (aref result row column)
            (* (aref matrix row column) scalar)))))

(defun sigmoid (x)
  (/ 1 (+ 1 (exp (- x)))))

(defun sigmoid-map (matrix)
  (matrix-map #'sigmoid matrix))

(defun sigmoid-derivative (x)
  (let ((v (sigmoid x)))
    (* v (- 1 v))))

(defun sigmoid-derivative-map (matrix)
  (matrix-map #'sigmoid-derivative matrix))

(defun output-n (weight-n activation-n-1 bias-n)
  (add-column bias-n (multiply weight-n activation-n-1)))

(defun activation-n (output-n)
  (sigmoid-map output-n))

(defun propogate (input weights-0 biases-0)
  (labels ((rec (weights biases outputs activations)
             (if (or weights biases)
               (let ((weight-n (first weights))
                     (activation-n-1 (first activations))
                     (bias-n (first biases)))
                 (let* ((output-n (output-n weight-n activation-n-1 bias-n))
                        (activation-n (activation-n output-n)))
                   (rec (rest weights)
                        (rest biases)
                        (cons output-n outputs)
                        (cons activation-n activations))))
               (values (nreverse outputs) (nreverse activations)))))
    (rec weights-0 biases-0 nil (list input))))

(defun cost (activation-l target)
  (scale (square (subtract activation-l target)) 0.5))

(defun nabla-l (output-l activation-l target)
  (hadamard (sigmoid-derivative-map output-l)
            (subtract activation-l target)))

(defun nabla-n (weights-1 outputs-0 nabla-l)
  (labels ((rec (weights outputs nablas)
             (if outputs
               (let ((weight-n+1 (first weights))
                     (output-n (first outputs))
                     (nabla-n+1 (first nablas)))
                 (rec (rest weights)
                      (rest outputs)
                      (cons
                        (hadamard
                          (multiply
                            (transpose weight-n+1)
                            nabla-n+1)
                          (sigmoid-derivative-map output-n))
                        nablas)))
               nablas)))
    (rec (reverse weights-1) (reverse outputs-0) (list nabla-l))))

(defun nablas (weights outputs activations target)
  (nabla-n (rest weights) 
           (butlast outputs) 
           (nabla-l (last1 outputs) (last1 activations) target)))

(defun weight-deltas (nablas-0 activations-0)
  (labels ((rec (nablas activations weight-deltas)
             (if nablas
               (let ((nabla-n (first nablas))
                     (activation-n-1 (first activations)))
                 (rec (rest nablas)
                      (rest activations)
                      (cons
                        (multiply
                          nabla-n
                          (transpose 
                            activation-n-1))
                        weight-deltas)))
               (nreverse weight-deltas))))
    (rec nablas-0 activations-0 nil)))

(defun bias-deltas (nablas-0)
  (loop for nablas in nablas-0
        collect (average-columns nablas)))

(defun update-weights (weight-deltas-0 weights-0 learning-rate)
  (loop for weight-deltas in weight-deltas-0
        for weights in weights-0
        collect (subtract weights (scale weight-deltas learning-rate))))

(defun update-biases (bias-deltas-0 biases-0 learning-rate)
  (loop for bias-deltas in bias-deltas-0
        for biases in biases-0
        collect (subtract biases (scale bias-deltas learning-rate))))

(defun random-matrix (rows columns)
  (iterate 
    (list rows columns)
    (lambda (result row column)
      (setf (aref result row column)
            (- 0.5 (random 1.0))))))

(defun random-weights (layer-sizes)
  (loop for size on layer-sizes
        while (second size)
        collect (random-matrix (second size) (first size))))

(defun random-biases (layer-sizes)
  (loop for size in (rest layer-sizes)
        collect (random-matrix size 1)))

(defun random-network (layer-sizes)
  (values (random-weights layer-sizes) (random-biases layer-sizes)))

;(defun cost (activation-l target)
(defun train (input target weights biases learning-rate desired-cost)
  (multiple-value-bind (outputs activations) (propogate input weights biases)
    (let ((cost (cost (last1 activations) target)))
      (if (> (aref (average-columns cost) 0 0) desired-cost)
        (let* ((nablas (nablas weights outputs activations target))
               (weight-deltas (weight-deltas nablas activations))
               (bias-deltas (bias-deltas nablas))
               (trained-weights (update-weights weight-deltas weights learning-rate))
               (trained-biases (update-biases bias-deltas biases learning-rate)))
          (train input target trained-weights trained-biases learning-rate desired-cost))
        (values weights biases)))))

;;;
;;; testing
;;;

;; xor
;(defvar input/  #2A((0 0 1 1)
;                    (1 0 1 0)))
;(defvar target/ #2A((1 0 0 1)))

;; and
(defvar input/  #2A((0 0 1 1)
                    (1 0 1 0)))
(defvar target/ #2A((0 0 1 0)))

(multiple-value-bind (weights biases) (random-network '(2 3 1))
  (defvar weights/ weights)
  (defvar biases/ biases))

(multiple-value-bind (trained-weights trained-biases) 
  (train input/ target/ weights/ biases/ 1.0 0.0001)
  (defvar trained-weights/ trained-weights)
  (defvar trained-biases/ trained-biases))

(multiple-value-bind (outputs activations) (propogate input/ trained-weights/ trained-biases/)
  (defvar outputs/ outputs)
  (defvar activations/ activations)
  (print-matricies outputs 'outputs)
  (print-matricies activations 'activations))
