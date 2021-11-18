import cv2

class SimpleMatcher:
    def __init__(self, matcher) -> None:
        self.matcher = matcher

    def __call__(self, descriptors):
        if self.matcher == 'normal':
            self.matches = self.normal(descriptors)
        elif self.matcher == 'knn':
            self.matches = self.knn(descriptors)
        elif self.matcher == 'flann':
            self.matches = self.flann(descriptors)
        else:
            raise TypeError('Wrong matcher type!')
        
        return self.matches

    def draw(self, image, keypoint):
        if self.matcher == 'normal':
            return cv2.drawMatches(
                image[0], keypoint[0],
                image[1], keypoint[1],
                self.matches[:100], None, flags=2)
        elif self.matcher == 'knn':
            return cv2.drawMatchesKnn(
                image[0], keypoint[0],
                image[1], keypoint[1],
                self.matches[:50], None, flags=2)
        elif self.matcher == 'flann':
            matchesMask = [[0, 0] for i in range(len(self.matches))]
            for i, (m, n) in enumerate(self.matches):
                if m.distance < 0.3*n.distance:
                    matchesMask[i] = [1, 0]

            draw_params = {
                'matchColor': (0, 255, 0),
                'singlePointColor': (255, 0, 0),
                'matchesMask': matchesMask,
                'flags': 0
            }
            return cv2.drawMatchesKnn(
                image[0], keypoint[0],
                image[1], keypoint[1],
                self.matches, None, **draw_params)
        else:
            raise TypeError('Wrong matcher type!')

    def normal(self, desc):
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        matches = bf.match(desc[0], desc[1])
        matches = sorted(matches, key = lambda x:x.distance)
        return matches

    def knn(self, desc):
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc[0], desc[1], k=2)
        # matches = sorted(matches, key = lambda x:x[0].distance - 0.3*x[1].distance)
        return matches

    def flann(self, desc):
        FLANN_INDEX_KDTREE = 0
        index_params = {'algorithm':FLANN_INDEX_KDTREE, 'trees':5}
        search_params = {'checks': 50}

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc[0], desc[1], k=2, compactResult=True)
        return matches